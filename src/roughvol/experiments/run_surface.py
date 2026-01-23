from __future__ import annotations

import argparse # argparse lets a Python script accept parameters from the command line in a clean, robust, and standard way.
import numpy as np

from roughvol.models.GBM_model import GBM_Model
from roughvol.instruments.vanilla import VanillaOption
from roughvol.engines.mc import MonteCarloEngine 

from roughvol.analytics.black_scholes_formula import implied_vol

'''
两个函数处理输入strike价格的格式处理和询问用户输入strike价格；main函数给出定价模型的输出结果。
'''

def parse_strikes(raw):
    strikes = []

    parts = raw.replace(",", " ").split()
    for p in parts:
        if ":" in p:
            # Range format: start:end:step
            try:
                start, end, step = map(float, p.split(":"))
                x = start
                while x <= end + 1e-12:
                    strikes.append(round(x, 10))
                    x += step
            except:
                raise ValueError(f"Invalid range format: {p}")
        else:
            strikes.append(float(p))

    return sorted(set(strikes))


def get_user_strikes():
    while True:
        raw = input(
            "\nEnter strikes (e.g. 80 90 100 or 80:120:5):\n> "
        )
        try:
            strikes = parse_strikes(raw)
            if len(strikes) == 0:
                raise ValueError("No strikes entered.")
            return strikes
        except Exception as e:
            print("Invalid input:", e)

'''
parse代表提取command-line中有意义的部分，即把string转换成所需变量的输入。
'''

def main():
    ap = argparse.ArgumentParser(description="BS Monte Carlo smile + implied vols.")
    ap.add_argument("--spot", type=float, default=100.0)
    ap.add_argument("--rate", type=float, default=0.02)
    ap.add_argument("--div", type=float, default=0.00)
    ap.add_argument("--vol", type=float, default=0.20)
    ap.add_argument("--maturity", type=float, default=1.0)
    ap.add_argument("--n_paths", type=int, default=200_000)
    ap.add_argument("--n_steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    
    # 输入商品模型
    model = GBM_Model(spot0=args.spot, rate=args.rate, div=args.div, vol=args.vol) 
    # 输入随机引擎
    engine = MonteCarloEngine(n_paths=args.n_paths, n_steps=args.n_steps, seed=args.seed, antithetic=True)

    strikes = get_user_strikes()

    print("strike, price, stderr, implied_vol")
    for k in strikes:
        opt = VanillaOption(strike=float(k), maturity=args.maturity, is_call=True)
        res = engine.price(model=model, instrument=opt)
        iv = implied_vol(price=res.price, spot=args.spot, strike=float(k), maturity=args.maturity,
                         rate=args.rate, div=args.div, is_call=True)
        print(f"{k:8.3f}, {res.price:12.6f}, {res.stderr:10.6f}, {iv:10.6f}")


'''
这个与main环境配对避免了意外的执行这个script，也可以import roughvol.experiments.run_surface，这样不回执行这个脚本。
'''

if __name__ == "__main__":
    main()
