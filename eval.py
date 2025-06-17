from transformers import DetrForObjectDetection, DetrConfig, DetrImageProcessor, Trainer, TrainingArguments
from torch.utils.data import random_split, ChainDataset
from dataset import DartsDataset
from torchvision import transforms
from PIL import Image, ImageDraw
import torch
from torchvision.datasets import CocoDetection
from argparse import ArgumentParser


image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", size={"shortest_edge": 768, "longest_edge": 768}) 

def collate_fn(batch):
    images, targets = zip(*batch)
    inputs = image_processor(images=images, annotations=targets, return_tensors="pt", do_resize=False)
    return inputs

if __name__ == "__main__":
    dirs = [f'3D/scene/rendered/imgs_{i}' for i in range(6)]# + 
    dataset = DartsDataset(dirs=dirs, random_rescale=True, random_rotation=True)
    real_dataset = DartsDataset(dirs=[f'my_unlabelled/vids/frames_000{i}' for i in range(8)], random_rescale=False, random_rotation=False, resize_to=768)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
    
    
    print("train/test split:", len(train_dataset), len(test_dataset))
    
    id2label = {
        0: "right",
        1: "top",
        2: "left",
        3: "bottom",
        4: "dart",
    }
    label2id = {v: k for k, v in id2label.items()}
    
    parser = ArgumentParser(description="eval")
    parser.add_argument("model_name", type=str, help="Path to model directory")
    args = parser.parse_args()
    
    model = DetrForObjectDetection.from_pretrained(
        args.model_name, 
        auxiliary_loss=True, 
        num_queries=100,
        num_labels=5,  
        # bbox_loss_coefficient=4.0,
        giou_loss_coefficient=0.0,
        # giou_cost=1.0,
        eos_coefficient = 0.06,
        ignore_mismatched_sizes=True,
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir="evaluation_results",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=30,
        # bf16=True,
        # max_grad_norm=.5,
        eval_strategy ="epoch",
        save_strategy ="epoch",
        logging_steps=50,
        learning_rate=2e-4,
        save_total_limit=5,
        remove_unused_columns=False,
        dataloader_num_workers = 2,
        dataloader_persistent_workers = True,
        lr_scheduler_type="cosine",
        metric_for_best_model="eval_test_loss", 
        # torch_compile = True,
        # torch_compile_backend = "inductor",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset={"test": test_dataset, "real": real_dataset},
        data_collator=collate_fn,
    )
    # trainer.train(resume_from_checkpoint=True)
    print(trainer.evaluate())

# only_synthetic/checkpoint-18000
# 'eval_test_loss': 28.642297744750977
# 'eval_real_loss': 9.77579116821289
# 'eval_test_loss': 12.114974021911621
# 'eval_real_loss': 2.7959001064300537

# multi_res_scratch_2/checkpoint-18000
# 'eval_test_loss': 5.276578426361084
# 'eval_real_loss': 15.685220718383789
# 'eval_test_loss': 1.733659267425537
# 'eval_real_loss': 6.519677639007568

# multi_res_scratch/checkpoint-18000
# 'eval_test_loss': 4.224990367889404
# 'eval_real_loss': 11.21245288848877
# 'eval_test_loss': 1.2451802492141724
# 'eval_real_loss': 3.466538429260254