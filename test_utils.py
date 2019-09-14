# This module cannot import any other PyTorch/XLA module. Only Python core modules.
import argparse
import sys


def parse_common_options(datadir=None,
                         num_cores=None,
                         batch_size=128,
                         num_epochs=10,
                         num_workers=4,
                         log_steps=20,
                         lr=None,
                         momentum=None,
                         opts=None):
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument('--datadir', type=str, default=datadir)
  parser.add_argument('--num_cores', type=int, default=num_cores)
  parser.add_argument('--batch_size', type=int, default=batch_size)
  parser.add_argument('--num_epochs', type=int, default=num_epochs)
  parser.add_argument('--num_workers', type=int, default=num_workers)
  parser.add_argument('--log_steps', type=int, default=log_steps)
  parser.add_argument('--lr', type=float, default=lr)
  parser.add_argument('--momentum', type=float, default=momentum)
  parser.add_argument('--metrics_debug', action='store_true')
  if opts:
    for name, aopts in opts:
      parser.add_argument(name, **aopts)
  args, leftovers = parser.parse_known_args()
  sys.argv = [sys.argv[0]] + leftovers
  return args
