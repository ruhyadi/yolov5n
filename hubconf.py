import os
import torch
import git



def _create(name: str = None, tag: str = None, classes: int = None, device: str = 'cpu'):

    from pathlib import Path
    from models.common import AutoShape
    from models.yolo import Model
    from utils.general import intersect_dicts

    # TODO: add your own weights list with extension
    weights_list = ['yolov5_nodeflux.pt']
    idx = [x.split('.')[0] for x in weights_list].index(name)
    assert name in [x.split('.')[0] for x in weights_list], \
        f'[INFO] not found, try: {weights_list}'

    repo = git.Repo(os.getcwd())
    username = (repo.remotes.origin.url).split('/')[-2]
    reponame = (repo.remotes.origin.url).split('/')[-1]
    repotag = tag
    if tag is None:
        repotag = repo.tags[-1] # last tag
    weights_url = f'https://github.com/{username}/{reponame}/releases/download/{repotag}/{weights_list[idx]}'    # https:/github.com/username/repo/releases/download/tags/model.pt
    # https:/github.com/ruhyadi/yolov5n/releases/download/v2.2/yolov5_nodeflux.pt

    # TODO: load your own model configuration
    cfg = list((Path(__file__).parent / 'models').rglob(f'{name}.yaml'))[0]  # model.yaml path
    model = Model(cfg, nc=classes)
    
    ckpt = torch.hub.load_state_dict_from_url(weights_url, map_location=device)
    csd = ckpt['model'].float().state_dict()
    csd = intersect_dicts(csd, model.state_dict(), exclude=['anchors'])
    model.load_state_dict(csd, strict=False)

    if len(ckpt['model'].names) == classes:
        model.names = ckpt['model'].names  # set class names attribute
    
    model = AutoShape(model)

    return model.to(device)

# TODO: define your model name
def yolov5_nodeflux(imgsize=640, classes=5, device='cpu'):
    # load custom model
    return _create('yolov5_nodeflux', tag='v2.0', device=device)
    
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description="Torch Hub Testing")
    parser.add_argument('--repository', type=str, required=True, help='Repository, username/reponame')
    parser.add_argument('--tag', type=str, required=True, help='Release tag')
    args = parser.parse_args()

    weights = torch.hub.list(f'{args.repository}:{args.tag}', force_reload=True)
    print(weights)
    for weight in weights:
        model = torch.hub.load(f'{args.repository}:{args.tag}', weight)
        print(f'[INFO] Success load {weight.upper()}')