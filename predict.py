import argparse
import torch
import utilities as util
import json

#image path: flowers/valid/20/image_04927.jpg

parser = argparse.ArgumentParser()
parser.add_argument('image_path', metavar='image_path', type=str, nargs=1)
parser.add_argument('checkpoint', metavar='checkpoint_path', type=str, nargs=1)
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--top_k', type=int)
parser.add_argument('--category_names', type=str)

if __name__ == '__main__':
    args = parser.parse_args()

    category_mapping = args.category_names if args.category_names else 'cat_to_name.json'
    with open(category_mapping, 'r') as f:
        cat_to_name = json.load(f)
        
    model = util.load_checkpoint(args.checkpoint)
    device = torch.device("cuda" if args.gpu else "cpu")
    print(device)
    image_path = args.image_path[0]
    topk = args.top_k if args.top_k else 1
    model.to(device)
    with torch.no_grad():
        model.eval()
        img = torch.from_numpy(util.process_image(image_path)).float().to(device)
        img.unsqueeze_(0)
        log_ps = model.forward(img)
    
        ps = torch.exp(log_ps)
        probs, classes = ps.topk(topk, 1)
        if device.type == 'cuda':
            probabilities, flowers = probs.cpu().numpy(), classes.cpu().numpy()
        else:
            probabilities, flowers = probs.numpy(), classes.numpy()
        
        for prob, flower in zip(probabilities[0], flowers[0]):
            print(f"Flower: {cat_to_name[str(flower)]}, Probability: {round(prob*100, 2)}%")