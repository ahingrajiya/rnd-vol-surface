import math
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Literal, NamedTuple

import numpy as np
from scipy.special import ndtr

from optionslib.core import (
    Market,
    Payoff,
)


@dataclass(frozen=True)
class NormalizedProblem:
    """
    A Black-Scholes inversion problem expressed in normalized coordinates

    In inverse BS problem starting with forward V = theta*D*[F*N(theta*d1)-K*N(theta*d2)]  d1=ln(F/K)/(sigma *sqrt(T))+1/2*sigma*sqrt(T) and d2= d1-sigma*sqrt(T).
    where theta = call_or_put, +1 for calls and -1 for puts. This leads to parameter collapse where F and K appears as ratio and sigma and T appears as
    product sigma*sqrt(T). Hence 4 parameters input (F,K,T,sigma) collapses to two parameters (x,s) where x is log moneyness, ln(F/K) and s is total volatility,
    sigma*sqrt(T). The pricing function's shape depends on (x,s), with D*sqrt(F*K) as an overall scale. The opposite convention, ln(K/F), is also in common
    use - the sign matters, since it selects the call and put branches.The Market provides (D,F) which is used to calculate (x,s).
    Normalized price is V/(D*sqrt(F*K))

    Attributes:
        T: Time to expiry in years. Must be positive
        K: Strike price. Must be positive
        F: Forward price. Must be positive
        D: Discount factor. Must be positive. Value of discount factor > 1 is permitted but it is not expected for the USD.
        D>1 is allowed in case this library is used to price markets where interest rates can be negative
        call_or_put : Corresponds to theta in the BS equation. Call takes +1 and put takes -1 value.

    Properties:
        x, sqrt_T and normalization_factor are properties derived from fields. They can not be inconsistent with the fields.

    Construct via:
        from_market: When you have Market, it reads D and F from it.
        Direct : When you have measured D and F to avoid inventing rate and dividend parameters in order to build a synthetic Market, only to recalculate F and D again.

    """

    T: float
    K: float
    F: float
    D: float
    call_or_put: Literal[-1, 1]

    def __post_init__(self):
        if self.K <= 0:
            raise ValueError(f"Strike should be positive, got value {self.K}")
        if self.F <= 0:
            raise ValueError(f"Forward should be positive, got value {self.F}")
        if self.call_or_put not in [-1, 1]:
            raise ValueError(
                f"Call or Put sign needs to be +1 or -1, respectively, got value {self.call_or_put}"
            )
        if self.T <= 0:
            raise ValueError(
                f"Square root of expiry has to be positive, got value {self.T}"
            )
        if self.D <= 0:
            raise ValueError(f"Discount factor should be positive, got value {self.D}")

    @property
    def x(self) -> float:
        """Log Moneyness ln(F/K)"""
        return np.log(self.F / self.K)

    @property
    def sqrt_T(self) -> float:
        return np.sqrt(self.T)

    @property
    def normalization_factor(self) -> float:
        return self.D * np.sqrt(self.K * self.F)

    @classmethod
    def from_market(
        cls, payoff: "Payoff", market: "Market", T: float
    ) -> "NormalizedProblem":
        return cls(
            D=market.discount_factor(T),
            F=market.forward(T),
            K=payoff.strike,
            call_or_put=payoff.call_or_put,
            T=T,
        )


class NormalizedBounds(NamedTuple):
    """
    Normalized price bounds

    Bounds are in the normalized price units b = V/(D*sqrt(F*K)), not dollars. The interval (floor,ceiling) is open at both ends. Floor is limit of b as s ->0+ and
    ceiling is limit of b as s-> infinity

    Attributes:
        floor : Floor bound for normalized price
        ceiling : Ceiling bound for normalized price

    """

    floor: float
    ceiling: float


def normalized_bounds(x: float, call_or_put: int) -> NormalizedBounds:
    """
    Calculates normalized bounds

    Floor is calculated as max(2*sinh(x_eff/2),0). Floor uses sinh over exponential differnce as near x = 0 difference would cancel and most liquid quotes are
    near x=0. Ceiling is calculated as exp(x_eff/2), wehre x_eff = x*call_or_put. Since Put and Call bounds are symmetric in x i.e. x->-x,
    we multiply call_or_put with x.

    Args:
        x (float): log moneyness
        call_or_put (int): +1 for call and -1 for puts

    Returns:
        NormalizedBounds: Normalized price bounds with floor and ceiling.
    """
    x_eff = x * call_or_put
    lower_bound = max(2 * math.sinh(x_eff / 2), 0)
    upper_bound = math.exp(x_eff / 2)

    return NormalizedBounds(floor=lower_bound, ceiling=upper_bound)


def normalized_price(x: float, call_or_put: int, s: float) -> float:
    """
    Calculates normalized price

    Normalized price is calculated as b = exp(x_eff/2)*N(x_eff/s+s/2)-exp(-x_eff/2)*N(x_eff/s-s/2), where x_eff = x*call_or_put and b = V/(D*sqrt(FK))
    Normalized price is symmetric with respect to log moneyness for puts and calls. The s is total volatility and has to be positive.
    Normalized price b will lose precision for large negative x_eff and small s due to both ndtr term being tiny and near equal.
    The better approach has been described in "Let's be rational" by Jäckel. Implementation of this work for this is currently out of scope.

    Args:
        x (float): log moneyness
        call_or_put (int): +1 for call and -1 for put options
        s (float): total volatility. Must be positive

    Returns:
        float: Returns normalized price
    """
    if s <= 0.0:
        raise ValueError(f"Total volatility s has to be positive, got value {s}")

    x_eff = x * call_or_put
    b = math.exp(x_eff / 2.0) * ndtr(x_eff / s + s / 2.0) - math.exp(
        -x_eff / 2.0
    ) * ndtr(x_eff / s - s / 2.0)
    return b


def normalized_vega(x: float, call_or_put: int, s: float) -> float:
    """
    Calculates normalized vega.

    Normalized Vega is the sensitivity of options price to the volatility. It is calculated as partial(b)/partial(s) = (1/sqrt(2*(pi)))*exp(-x^2/(2s^2)-s^2/8).
    Vega is defined as partial(V)/partial(sigma). Vega is related to normalized vega by vega  = (normalized_vega)*D*sqrt(FK)*sqrt(T). For vanilla European
    options (puts and calls), Vega is strictly positive and peaks near the money. Normalized Vega peaks at s_c = sqrt(2*abs(x)).
    Vega (or normalized vega) is the same for both put and call options, since x enters only as x^2. call_or_put is accepted but unused. It is kept in the signature so
    every function in this module has the same call shape.


    Args:
        x (float): log Moneyness
        call_or_put (int): +1 for call and -1 for puts. Plays no role in vega.
        s (float): Total volatility

    Returns:
        float: Returns normalized vega
    """
    if s <= 0.0:
        raise ValueError(f"Total volatility s has to be positive, got value {s}")

    del call_or_put  # call_or_put is not needed and this line removes linter error of unused call_or_put
    vega = (1.0 / (math.sqrt(2 * math.pi))) * math.exp(
        -(x**2) / (2.0 * s**2) - s**2 / 8.0
    )

    return vega


class Reason(StrEnum):
    """
    Gives reason for the convergence or failure of solver

    Attributes:
        OK : Solver converged with a unique solution
        BELOW_FLOOR/ABOVE_CEILING : Drop the quote as it violates no arbitrage condition
        ROOT_BELOW_BRACKET : Bracket for volatility is wrong. Widen the bottom of your bracket
        ROOT_ABOVE_BRACKET : Bracket for volatility is wrong. Widen the top of your bracket
        NO_CONVERGENCE : The solver failed to converge within iteration cap
    """

    @staticmethod
    def _generate_next_value_(
        name: str, start: int, count: int, last_values: list[str]
    ) -> str:
        return name

    OK = auto()
    BELOW_FLOOR = auto()
    ABOVE_CEILING = auto()
    ROOT_BELOW_BRACKET = auto()
    ROOT_ABOVE_BRACKET = auto()
    NO_CONVERGENCE = auto()


class Method(StrEnum):
    """
    Gives which method produced result

    Attributes:
        NEWTON : Newton Raphson method converged in given iteration cap
        BRENT : Newton Raphson failed (iteration cap or step outside the bracket) and Brent method converged in given iteration cap
        NONE : No method ran because bounds or bracket check rejected quote
    """

    @staticmethod
    def _generate_next_value_(
        name: str, start: int, count: int, last_values: list[str]
    ) -> str:
        return name

    NEWTON = auto()
    BRENT = auto()
    NONE = auto()


@dataclass(frozen=True)
class ImpliedVolResult:
    """
    ImpliedVolResult holds different results from the implied volatility solver.

    Attributes:
        reason : Gives the reason for failure of solver. In case of success it holds value OK
        x : Log Moneyness
        floor : The floor bound for normalized price
        ceiling : The ceiling bound for normalized price
        sigma : The calculated implied volatility.
        method : which method was used by solver for the implied volatility calculation
        s : total volatility, s = sigma*sqrt(T) at the root. The root on success, last iterate on NO_CONVERGENCE, NaN otherwise
        iterations : number of iterations performed to find solution

    Note :
        x, floor and ceiling are populated on every path so failure can be binned by moneyness without recomputation
    """

    reason: Reason
    x: float
    floor: float
    ceiling: float
    sigma: float = float("nan")
    method: Method = Method.NONE
    s: float = float("nan")
    iterations: int = 0

    def __post_init__(self):
        if self.reason != Reason.OK and not math.isnan(self.sigma):
            raise ValueError(
                f"Result did not converge but volatility is not NaN, got value{self.sigma}"
            )


def bounds_check(
    b_target: float, floor: float, ceiling: float, epsilon: float = 1e-12
) -> Reason:
    """
    Checks the normalized price bounds and returns the reason
    Price bound is from open interval (floor, ceiling). A parameter epsilon is used for strict comparison for an open interval. Normalized price is expected
    to be of the order of 0.001 to 1.0 in most cases. Hence defaulting epsilon = 1e-12 is comfortably above floating point noise. However, it is passed as argument
    in case of noisy data requires loosening of it.

    Args:
        b_target (float): normalized price
        floor (float): floor limit of normalized price
        ceiling (float): ceiling limit of normalized price
        epsilon (float, optional): interval tolerance. Defaults to 1e-12.

    Returns:
        Reason: OK for proceed or reason for bound failure
    """
    if b_target <= floor + epsilon:
        return Reason.BELOW_FLOOR
    if b_target >= ceiling - epsilon:
        return Reason.ABOVE_CEILING
    return Reason.OK


@dataclass(frozen=True)
class SolverConfig:
    s_lo: float = 0.05
    s_hi: float = 5.0
    epsilon: float = 1e-12
    max_newton_iterations: int = 8
    max_brent_iterations: int = 32
    tolerance: float = 1e-8

    def __post_init__(self):
        if self.s_lo <= 0:
            raise ValueError(
                f"Total implied vol lower bound needs to be positive, got value {self.s_lo}"
            )
        if self.s_hi <= 0:
            raise ValueError(
                f"Total implied vol upper bound needs to be positive, got value {self.s_hi}"
            )
        if self.epsilon <= 0:
            raise ValueError(
                f"Bounds check tolerance needs to be positive, got value {self.epsilon}"
            )
        if self.max_newton_iterations <= 0:
            raise ValueError(
                f"Maximum iteration for convergence using Netwon method needs to be positive, got value {self.max_newton_iterations}"
            )
        if self.max_brent_iterations <= 0:
            raise ValueError(
                f"Maximum iteration for convergence using Brent method needs to be positive, got value {self.max_brent_iterations}"
            )
        if self.tolerance <= 0:
            raise ValueError(
                f"Convergence tolerance of |delta s| needs to be positive, got value {self.tolerance}"
            )
        if self.s_lo >= self.s_hi:
            raise ValueError(
                f"Lower bound for total implied vol needs to be smaller than upper bound, got lower bound = {self.s_lo} >= upper bound {self.s_hi}"
            )
