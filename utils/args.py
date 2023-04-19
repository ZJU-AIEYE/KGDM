import argparse
def get_args(handle_args=None):
    parser = argparse.ArgumentParser()
    # Distributed and accelerate setting
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--use_amp', action="store_true")

    # Training setting
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--resume', action="store_true")
    parser.add_argument('--expid', type=str, default='test')
    parser.add_argument('--data', type=str, default = '')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--evidence', type=str, default='softplus')
    parser.add_argument('--num_fold', type=int, default=5)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', action="store_true")
    parser.add_argument('--logdir', type=str, default='outputs')
    parser.add_argument('--save_model_dir', type=str, default='/data/home/fangzhengqing/Result')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--classifier', type=str, default='DGS')
    parser.add_argument('--init_weight', type=str, default='default')
    parser.add_argument('--proto_dim', type=int, default=128)


    # Optimizer and scheduler setting

    parser.add_argument('--kl_coef', type=float, default=10)
    parser.add_argument('--op_type', type=str, default="Adam")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta0', type=float, default=0.9)
    parser.add_argument('--beta1', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--milestones', nargs='+', type=int, default=[20, 40])
    parser.add_argument('--lr_decay', type=float, default=0.1)
    
    if handle_args:
        args = parser.parse_args(handle_args)
    else:
        args = parser.parse_args()
    
    return args