from rfdetr import RFDETRBase, RFDETRSmall, RFDETRMedium, RFDETRLarge
import argparse
import clearml

def train_rf_detr(model_size='base',epochs=20,dataset_name="/mnt/fish-detection-coco-dataset/benthic_f_small/rf_data",
                    output_uri="/mnt/fish-detection-coco-dataset/benthic_f_small/models/rf_detr10",dataset_id=None):
    #changes to rf-detr: coco.py line 316 res.dataset['info'] = {}
    batch_size = 16
    grad_accum_steps = 1
    if model_size == 'base':
        model = RFDETRBase()
    elif model_size == 'small':
        model = RFDETRSmall()
    elif model_size == 'medium':
        model = RFDETRMedium()
    elif model_size == 'large':
        batch_size = 8
        grad_accum_steps = 2
        model = RFDETRLarge()
    
    task = clearml.Task.init(project_name="RF-DETR_Benthic", task_name=f"Train RF-DETR {model_size} model",output_uri=output_uri)
    dataset = clearml.Dataset.get(dataset_id=dataset_id)
    dataset_path = dataset.get_local_copy()
    model.train(
        dataset_dir=dataset_path,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        lr=1e-4,
        output_dir="./output",
        # resume= "/mnt/fish-detection-coco-dataset/benthic_f_small/models/rf_detr10/checkpoint.pth",
        checkpoint_interval=1,
        tensorboard = True,

    )
# tsp python training.py --model_size base --epochs 20 --dataset_dir /mnt/fish-detection-coco-dataset/benthic_f_small/rf_data --output_dir /mnt/fish-detection-coco-dataset/benthic_f_small/models/rf_base20
# tsp python training.py --model_size medium --epochs 20 --dataset_dir /mnt/fish-detection-coco-dataset/benthic_f_small/rf_data_comb --output_dir /mnt/fish-detection-coco-dataset/benthic_f_small/models/rf_med_comb20
# tsp python training.py --model_size medium --epochs 20 --dataset_dir /mnt/fish-detection-coco-dataset/equinor_coco/rf_data_eq_all --output_dir /mnt/fish-detection-coco-dataset/equinor_coco/models/rf_med_eq20
if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description='Train RF-DETR model')
    argparse.add_argument('--model_size',type=str,help="Model size: base, small, medium, large",default='small')
    argparse.add_argument('--epochs',type=int,help="Number of training epochs",default=1)
    argparse.add_argument('--dataset_name',type=str,help="Path to dataset name",default=("RF-DETR_Benthic","benthic_s_comb","1.0.1"))
    argparse.add_argument('--dataset_id',type=str,help="Dataset ID",default="e6522f92a45f4cbb89095f22b81548c9")
    argparse.add_argument('--output_uri',type=str,help="Path to output directory for saving model checkpoints",default="/mnt/fish-detection-coco-dataset/")
    args = argparse.parse_args()
    train_rf_detr(model_size=args.model_size,epochs=args.epochs,dataset_name=args.dataset_name,output_uri=args.output_uri,dataset_id=args.dataset_id)    