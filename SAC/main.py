from sac import sac
from utils import parse_args

if __name__ == '__main__':
    args = parse_args()

    print(f"Creating Environment {args.env_name}")
    # env = gym.make(args.env_name)
    # env = RFactor2Environment()

    sac(skip_initial=args.skip_initial,
        alpha=args.alpha, gamma=args.gamma, tau=args.tau, lr=args.lr,
        replay_size=args.replay_size, batch_size=args.batch_size,
        epochs=args.epochs, start_steps=args.start_steps)
