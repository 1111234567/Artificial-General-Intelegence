"""Born-symmetric recurrent gate weights; same GRU training stream and budget."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone

import torch
from torch import nn
from torch.nn.utils import parametrize

from experiments.learn_remember import gru_train as plain


OUT=plain.ROOT/"data/learn-remember/symmetric-v1"


class SharedPairs(nn.Module):
    """Expand three independent unordered-pair tables, one per GRU gate."""
    def __init__(self, width):
        super().__init__()
        i,j=torch.triu_indices(width,width)
        index=torch.empty(width,width,dtype=torch.long)
        index[i,j]=index[j,i]=torch.arange(i.numel())
        self.register_buffer("index",index)
        self.register_buffer("i",i)
        self.register_buffer("j",j)

    def forward(self, packed):
        width=self.index.shape[0]
        return packed[:,self.index].reshape(3*width,width).contiguous()

    def right_inverse(self, dense):
        width=self.index.shape[0]
        return dense.reshape(3,width,width)[:,self.i,self.j].contiguous()


def constrain(model):
    rule=SharedPairs(model.width).to(model.head.weight.device)
    parametrize.register_parametrization(model.core,"weight_hh_l0",rule,unsafe=True)
    return model


def deploy(model):
    # Copy only learned values. Avoid consuming any training random draws when
    # constructing the temporary public controller for collection/evaluation.
    with torch.random.fork_rng(devices=[]):
        public=plain.GRUAgent(**model.config()).to(dtype=model.head.weight.dtype)
    sd=dict(model.state_dict())
    packed=model.core.weight_hh_l0.detach().reshape(3,model.width,model.width)
    assert torch.equal(packed,packed.transpose(-1,-2))
    sd={k:v for k,v in sd.items() if ".parametrizations." not in k}
    sd["core.weight_hh_l0"]=packed.reshape(3*model.width,model.width)
    public.load_state_dict(sd)
    return public


class SharedTrainer(plain.Trainer):
    def __init__(self,cfg,folder):
        super().__init__(cfg)
        self.folder=folder
        self.folder.mkdir(parents=True,exist_ok=False)
        self.model=constrain(self.model)
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=cfg.lr)

    def collect(self, teacher_probability):
        training=self.model
        self.model=deploy(training)
        try:
            return super().collect(teacher_probability)
        finally:
            self.model=training

    def save(self,name):
        training=self.model
        self.model=deploy(training)
        try:
            super().save(name)
        finally:
            self.model=training
        plain.write_json(self.folder/(name+".symmetry.json"),dict(
            stage=name,trainable_parameters=sum(p.numel() for p in training.parameters()),
            unique_hidden_coordinates=3*training.width*(training.width+1)//2,
            archived_dense_parameters=sum(p.numel() for p in training.parameters())+3*training.width*(training.width-1)//2,
            packed_from_dense="right_inverse reconstitutes the exact constrained trainable coordinates",
            checkpoint_sha256=plain.sha(self.folder/(name+".pt")),source_sha256=plain.sha(__file__)))


def test_shared_gradients():
    torch.manual_seed(902)
    model=constrain(plain.GRUAgent(width=8)).double()
    public=deploy(model)
    obs=torch.randn(11,3,56,dtype=torch.float64)
    h=torch.randn(3,8,dtype=torch.float64)
    native,_=public(obs,h); actual,_=model(obs,h)
    assert torch.allclose(native,actual,atol=1e-12,rtol=1e-12)
    actual.square().sum().backward(); native.square().sum().backward()
    rule=model.core.parametrizations.weight_hh_l0[0]
    grad=public.core.weight_hh_l0.grad.reshape(3,8,8)
    ref=grad[:,rule.i,rule.j]+grad[:,rule.j,rule.i]*(rule.i!=rule.j)
    assert torch.allclose(ref,model.core.parametrizations.weight_hh_l0.original.grad,atol=1e-12)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("declare","test","train"))
    parser.add_argument("--seed",type=int,choices=range(3))
    args=parser.parse_args()
    cfg=plain.Config(seed=args.seed or 0)
    if args.command=="test":
        test_shared_gradients()
        print("symmetric expansion, dense controller and pair derivatives passed")
    elif args.command=="declare":
        OUT.mkdir(exist_ok=True,parents=True)
        path=OUT/"protocol.json"
        if path.exists():
            raise FileExistsError(path)
        plain.write_json(path,dict(created_utc=datetime.now(timezone.utc).isoformat(),config=asdict(cfg),
            seeds=[0,1,2],source_sha256=plain.sha(__file__),parent_sha256=plain.sha(plain.__file__),
            origin="Posthoc transfer of the promising plain GRU curriculum to reciprocal hidden couplings; fixed before examining any shared-pair trial results.",
            hypothesis="Can a recurrent learner born with trainable shared weights for each unordered hidden node pair acquire and retain the same long task?",
            definition="Three independent symmetric recurrent matrices, one per GRU gate. Diagonal self-weights are learned. Dense input and action interfaces are directed; the node-dependent gates need not give reciprocal effective influences.",
            controls="Plain trainer's exact public data generator, curriculum, model width, seed, optimizer, update counts, memory and no-replay withdrawal. Upper triangle copied from the same initial random draw, mirrored from birth. Different self-generated actions can change later observations/labels.",
            evaluation="Same independent common heldout test worlds and forced-history memory assay as plain. No selection of favorable seeds. Compare end of acquisition and no-replay withdrawal, not a cost-matched comparison to the 80-node graph."))
    else:
        import json
        plan=json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
        if plan["source_sha256"]!=plain.sha(__file__) or plan["parent_sha256"]!=plain.sha(plain.__file__):
            raise ValueError("training source changed")
        learner=SharedTrainer(cfg,OUT/f"seed_{cfg.seed:03d}")
        learner.run()


if __name__=="__main__":
    main()
