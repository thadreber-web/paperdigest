import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention_is_all_you_need._config import load_config
from attention_is_all_you_need.tracking import log_run
from attention_is_all_you_need.transformer import Transformer
from attention_is_all_you_need.layers import feed_forward, layer_norm, residual
from attention_is_all_you_need.position import positional_encoding

def main():
    parser = argparse.ArgumentParser(description='Train Transformer model')
    parser.add_argument('--config', required=True, help='Path to config file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Build model
    model = Transformer(
        d_model=config.d_model,
        N=config.N,
        h=config.h,
        d_ff=config.d_ff,
        P_drop=config.P_drop,
        d_k=config.d_k,
        d_v=config.d_v
    )
    
    # Training placeholder
    model.train()
    
    log_run('train', {
        'd_model': config.d_model,
        'N': config.N,
        'h': config.h,
        'd_ff': config.d_ff,
        'P_drop': config.P_drop
    })

if __name__ == '__main__':
    main()
