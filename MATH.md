# Part 1: Implied Volatility Derivation
In Black-Scholes options pricing formula, volatility is the unknown. One way to caulculate volatility is to invert the Black-Scholes formula using provided forward price. The calculated volatility is called implied volatility. In this part, we derive the mathematics needed to invert the Black-Scholes formula to calculate the implied volatility. The following notations will be used:

| Symbol | Meaning |
|---|---|
| $F$ | Forward price for delivery at $T$ |
| $K$ | Strike price |
| $T$ | Time to expiry in years |
| $D$ | Discount factor to expiry |
| $C$ | Call price |
| $P$ | Put price |
| $\sigma$ | Annualized volatility |
| $s = \sigma\sqrt{T}$ | Total volatility |
| $x = \ln(F/K)$ | Log moneyness.|
| $N$ | Normal distribution function |


$x < 0$ indicates out-of-the-money calls (in-the-money puts) and $x > 0$ indicates in-the-money calls (out-of-the-money puts) 
## 1.1 The forward Black-Scholes price 
