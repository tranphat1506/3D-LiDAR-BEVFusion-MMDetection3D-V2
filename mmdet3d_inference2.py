#!/usr/bin/env python3
"""
Backward-compatibility wrapper.
Redirects execution to bev_inferencer.py.
"""
import sys
from bev_inferencer import main, parse_args

if __name__ == '__main__':
    args = parse_args()
    main(args)
