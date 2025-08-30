#!/usr/bin/env python3
"""
Batch Auction Mathematical Engine - CORRECTED VERSION
====================================================

Complete implementation of the mathematical framework for discrete-time batch auctions
with risk-aware clearing using Quadratic Programming.

Mathematical Corrections Applied:
1. Fixed objective function sign consistency
2. Corrected KKT condition analysis
3. Enhanced numerical stability
4. Proper constraint formulation
5. Improved regularization handling

Features:
- Piecewise-linear order curves
- Quadratic risk model with covariance matrix
- Welfare maximization with inventory risk penalties
- KKT condition analysis
- Real-time price discovery (with optional price smoothing)

Usage: python batch_auction.py [--interactive | --test {standard,stress,custom,multi}] [--config FILE] [--verbose]
"""

import numpy as np
import json
import argparse
import sys
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Optional SciPy import guarded so the file can be imported where SciPy isn't present
try:
    from scipy.optimize import minimize
    from scipy.linalg import cholesky, LinAlgError
except Exception as _e:  # pragma: no cover
    minimize = None
    cholesky = None
    class LinAlgError(Exception): ...
    _SCIPY_IMPORT_ERROR = _e
else:
    _SCIPY_IMPORT_ERROR = None


class TokenType(Enum):
    USDC = 0
    SOL = 1
    ETH = 2
    BTC = 3


@dataclass
class OrderPiece:
    """Single piece of a piecewise-linear order curve"""
    quantities: np.ndarray  # x_j: quantity vector (positive=buy, negative=sell)
    value: float            # v_j: reservation value in numéraire
    user_id: str            # identifier
    piece_id: int           # piece number for this user

    def __post_init__(self):
        self.quantities = np.array(self.quantities, dtype=float)


@dataclass
class MarketState:
    """Current market state and configuration"""
    num_tokens: int
    numeraire_idx: int = 0  # USDC index
    inventory: np.ndarray = None
    target_inventory: np.ndarray = None
    covariance_matrix: np.ndarray = None
    risk_aversion: float = 1.0
    inventory_bounds: Tuple[np.ndarray, np.ndarray] = None

    def __post_init__(self):
        if self.inventory is None:
            self.inventory = np.zeros(self.num_tokens)
        if self.target_inventory is None:
            self.target_inventory = np.zeros(self.num_tokens)
        if self.covariance_matrix is None:
            # Default to diagonal covariance with unit variance
            self.covariance_matrix = np.eye(self.num_tokens)
        if self.inventory_bounds is None:
            self.inventory_bounds = (
                np.full(self.num_tokens, -np.inf),
                np.full(self.num_tokens, np.inf)
            )
        if self.risk_aversion <= 0:
            raise ValueError("Risk aversion must be positive")
        if not np.allclose(self.covariance_matrix, self.covariance_matrix.T):
            raise ValueError("Covariance matrix must be symmetric")
        
        # Convert arrays to proper types
        self.inventory = np.array(self.inventory, dtype=float)
        self.target_inventory = np.array(self.target_inventory, dtype=float)
        self.covariance_matrix = np.array(self.covariance_matrix, dtype=float)


class BatchAuctionEngine:
    """Core mathematical engine for batch auction clearing"""

    def __init__(self, market_state: MarketState):
        if _SCIPY_IMPORT_ERROR is not None:
            raise RuntimeError(
                "SciPy is required to run the solver parts of this engine. "
                f"Import error: {_SCIPY_IMPORT_ERROR}"
            )
        self.market = market_state
        self.pieces: List[OrderPiece] = []
        self.solution: Optional[Dict] = None
        self.prev_prices: Optional[np.ndarray] = None

        # Validate and compute Cholesky decomposition
        try:
            # Check if matrix is positive definite first
            eigenvals = np.linalg.eigvals(self.market.covariance_matrix)
            if np.min(eigenvals) <= 1e-12:
                print(f"Warning: Covariance matrix has small/negative eigenvalues: min = {np.min(eigenvals):.2e}")
                reg_cov = self.market.covariance_matrix + 1e-6 * np.eye(self.market.num_tokens)
                self.chol_L = cholesky(reg_cov, lower=True)
            else:
                self.chol_L = cholesky(self.market.covariance_matrix, lower=True)
        except LinAlgError:
            print("Warning: Covariance matrix not positive definite. Using regularized version.")
            reg_cov = self.market.covariance_matrix + 1e-6 * np.eye(self.market.num_tokens)
            self.chol_L = cholesky(reg_cov, lower=True)

    def add_order_piece(self, piece: OrderPiece):
        """Add a single piece to the order book"""
        if len(piece.quantities) != self.market.num_tokens:
            raise ValueError(f"Order quantities must have length {self.market.num_tokens}")
        self.pieces.append(piece)
        print(f"Added piece: User {piece.user_id}, Piece {piece.piece_id}")
        print(f"  Quantities: {piece.quantities}")
        print(f"  Value: {piece.value}")

    def add_limit_order(self, user_id: str, token_idx: int, quantity: float,
                        limit_price: float, is_buy: bool = True):
        """Add a simple limit order (single piece) - CORRECTED"""
        if token_idx < 0 or token_idx >= self.market.num_tokens:
            raise ValueError(f"Token index must be between 0 and {self.market.num_tokens-1}")
        if token_idx == self.market.numeraire_idx:
            raise ValueError("Cannot trade the numeraire token directly")
        
        quantities = np.zeros(self.market.num_tokens)

        if is_buy:
            # Buy order: receive token, pay numeraire
            quantities[token_idx] = abs(quantity)  
            quantities[self.market.numeraire_idx] = -abs(quantity * limit_price)  
            # Value for buy order is the utility gained (can be set to the limit price * quantity for simple case)
            value = abs(quantity * limit_price)
        else:
            # Sell order: give token, receive numeraire
            quantities[token_idx] = -abs(quantity)  
            quantities[self.market.numeraire_idx] = abs(quantity * limit_price)  
            # Value for sell order is the utility from receiving numeraire
            value = abs(quantity * limit_price)

        piece = OrderPiece(quantities, value, user_id, len(self.pieces))
        self.add_order_piece(piece)

    def add_order_ladder(self, user_id: str, token_idx: int,
                         quantities: List[float], prices: List[float], is_buy: bool = True):
        """Add a ladder order (multiple price levels)"""
        if len(quantities) != len(prices):
            raise ValueError("Quantities and prices must have same length")
        for i, (qty, price) in enumerate(zip(quantities, prices)):
            self.add_limit_order(f"{user_id}_ladder_{i}", token_idx, qty, price, is_buy)

    def compute_risk(self, inventory: np.ndarray) -> float:
        """Compute quadratic risk: 0.5 * (q - q_bar)^T * Σ * (q - q_bar)"""
        deviation = inventory - self.market.target_inventory
        # Use Cholesky decomposition for numerical stability: ||L^T * deviation||^2
        chol_dev = self.chol_L.T @ deviation
        return 0.5 * np.dot(chol_dev, chol_dev)

    def compute_risk_gradient(self, inventory: np.ndarray) -> np.ndarray:
        """Compute gradient of risk function: Σ * (q - q_bar)"""
        deviation = inventory - self.market.target_inventory
        return self.market.covariance_matrix @ deviation

    def solve_batch_auction(self, verbose: bool = True,
                            price_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                            regularization: float = 1e-8,
                            price_smoothing: float = 1e-6) -> Dict:
        """
        Solve the batch auction QP - CORRECTED FORMULATION:

        max_{α,p}  Σ α_j v_j - λ * 0.5 * (q' - q̄)^T Σ (q' - q̄) - η * ||p - p_prev||²

        s.t.  0 ≤ α_j ≤ 1
              p_min ≤ p ≤ p_max, p_numeraire = 1
              p^T * (X^T α) = 0  (budget constraint - CORRECTED to equality)
              q_min ≤ q' ≤ q_max
        """
        if minimize is None:
            raise RuntimeError("SciPy is required to run the solver.")

        if not self.pieces:
            return {"status": "no_orders", "message": "No orders to clear"}

        n_pieces = len(self.pieces)
        n_tokens = self.market.num_tokens
        n_vars = n_pieces + n_tokens  # α variables + price variables

        # Default price bounds (prevent degenerate prices)
        if price_bounds is None:
            pmin = np.full(n_tokens, 0.001)
            pmax = np.full(n_tokens, 10000.0)
            pmin[self.market.numeraire_idx] = 1.0
            pmax[self.market.numeraire_idx] = 1.0
            price_bounds = (pmin, pmax)

        # Extract order data
        X = np.array([piece.quantities for piece in self.pieces])  # n_pieces × n_tokens
        v = np.array([piece.value for piece in self.pieces])       # n_pieces

        if verbose:
            print(f"\nSolving auction with {n_pieces} pieces and {n_tokens} tokens")
            print(f"Order matrix X shape: {X.shape}")
            print(f"Values v: {v}")

        def objective(vars_):
            """CORRECTED: Minimize negative welfare + risk + regularization"""
            alpha = vars_[:n_pieces]
            prices = vars_[n_pieces:]

            # Aggregate flow
            aggregate_flow = X.T @ alpha  # n_tokens vector
            new_inventory = self.market.inventory + aggregate_flow

            # Risk term - using efficient Cholesky computation
            deviation = new_inventory - self.market.target_inventory
            chol_dev = self.chol_L.T @ deviation
            risk = 0.5 * np.dot(chol_dev, chol_dev)

            # Welfare term (negative because we're minimizing)
            welfare = -np.dot(alpha, v)

            # Regularization term on alpha (prevent extreme solutions)
            reg_alpha = regularization * np.sum(alpha ** 2)

            # Price smoothing (if previous prices available)
            price_smooth = 0.0
            if self.prev_prices is not None and price_smoothing > 0:
                # Only smooth non-numeraire prices
                price_diff = prices - self.prev_prices
                price_diff[self.market.numeraire_idx] = 0.0  # Don't penalize numeraire
                price_smooth = price_smoothing * np.sum(price_diff ** 2)

            # Total objective (to minimize)
            total_obj = welfare + self.market.risk_aversion * risk + reg_alpha + price_smooth
            return total_obj

        def constraint_budget_eq(vars_):
            """CORRECTED: Budget constraint as equality: p^T (X^T α) = 0"""
            alpha = vars_[:n_pieces]
            prices = vars_[n_pieces:]
            aggregate_flow = X.T @ alpha
            budget_value = np.dot(prices, aggregate_flow)
            return budget_value  # = 0

        def constraint_inventory_bounds(vars_):
            """Combined inventory bounds constraint"""
            alpha = vars_[:n_pieces]
            aggregate_flow = X.T @ alpha
            new_inventory = self.market.inventory + aggregate_flow
            
            # Return vector of constraint violations (should be >= 0)
            q_min, q_max = self.market.inventory_bounds
            lower_violations = new_inventory - q_min  # >= 0
            upper_violations = q_max - new_inventory  # >= 0
            
            return np.concatenate([lower_violations, upper_violations])

        # Initial guess - IMPROVED
        x0 = np.zeros(n_vars)
        # Start with moderate fills to avoid corner solutions
        x0[:n_pieces] = 0.1  
        # Initialize prices based on simple supply/demand heuristic
        if len(self.pieces) > 0:
            # Estimate prices from order flow
            total_flow = np.sum(X, axis=0)
            for i in range(n_tokens):
                if i != self.market.numeraire_idx:
                    # Simple price estimate based on flow direction
                    if total_flow[i] > 0:  # Net buying
                        x0[n_pieces + i] = 1.1  # Slightly above par
                    elif total_flow[i] < 0:  # Net selling  
                        x0[n_pieces + i] = 0.9  # Slightly below par
                    else:
                        x0[n_pieces + i] = 1.0
        
        x0[n_pieces:] = np.maximum(x0[n_pieces:], price_bounds[0])
        x0[n_pieces:] = np.minimum(x0[n_pieces:], price_bounds[1])
        x0[n_pieces + self.market.numeraire_idx] = 1.0  # Numeraire price = 1

        # Bounds
        bounds = []
        for _ in range(n_pieces):
            bounds.append((0.0, 1.0))  # alpha bounds
        for i in range(n_tokens):
            if i == self.market.numeraire_idx:
                bounds.append((1.0, 1.0))
            else:
                bounds.append((price_bounds[0][i], price_bounds[1][i]))

        # Constraints - CORRECTED
        constraints = [
            {'type': 'eq', 'fun': constraint_budget_eq},  # CHANGED: equality constraint
            {'type': 'ineq', 'fun': constraint_inventory_bounds},
        ]

        # Multiple solver attempts for robustness
        methods = ['SLSQP', 'trust-constr']
        result = None
        
        for method in methods:
            try:
                if verbose:
                    print(f"Attempting with method: {method}")
                
                # Method-specific options
                if method == 'SLSQP':
                    options = {'maxiter': 1000, 'ftol': 1e-9, 'disp': False}
                else:  # trust-constr
                    options = {'maxiter': 1000, 'gtol': 1e-8, 'disp': False}
                
                result = minimize(
                    objective, x0, method=method, bounds=bounds, constraints=constraints,
                    options=options
                )
                
                if result.success:
                    # Verify solution quality
                    budget_error = abs(constraint_budget_eq(result.x))
                    if budget_error < 1e-6:  # Good solution
                        break
                    else:
                        if verbose:
                            print(f"Solution has large budget error: {budget_error:.2e}")
                        
            except Exception as e:
                if verbose:
                    print(f"Method {method} failed: {e}")
                continue

        # Fallback: relaxed problem if exact budget constraint fails
        if result is None or not result.success or abs(constraint_budget_eq(result.x)) > 1e-5:
            if verbose:
                print("Attempting with relaxed budget constraint...")

            def constraint_budget_relaxed(vars_):
                """Allow small budget violations"""
                alpha = vars_[:n_pieces]
                prices = vars_[n_pieces:]
                aggregate_flow = X.T @ alpha
                budget_value = np.dot(prices, aggregate_flow)
                # Allow small positive spending (≤ 1e-4)
                return 1e-4 - budget_value

            constraints_relaxed = [
                {'type': 'ineq', 'fun': constraint_budget_relaxed},
                {'type': 'ineq', 'fun': constraint_inventory_bounds},
            ]

            result = minimize(
                objective, x0, method='SLSQP', bounds=bounds,
                constraints=constraints_relaxed,
                options={'maxiter': 2000, 'ftol': 1e-9}
            )

        if not result.success:
            return {
                "status": "failed",
                "message": getattr(result, "message", "Unknown solver error"),
                "infeasible_reason": self._diagnose_infeasibility(X, v, verbose)
            }

        # Extract solution
        alpha_star = result.x[:n_pieces]
        prices_star = result.x[n_pieces:]
        aggregate_flow = X.T @ alpha_star
        new_inventory = self.market.inventory + aggregate_flow

        # Compute final metrics
        welfare = np.dot(alpha_star, v)
        risk = self.compute_risk(new_inventory)
        objective_value = welfare - self.market.risk_aversion * risk

        # Verify solution constraints
        budget_error = np.dot(prices_star, aggregate_flow)
        max_alpha = np.max(alpha_star)
        min_alpha = np.min(alpha_star)

        if verbose:
            print(f"\n✅ Solution verification:")
            print(f"  Budget error: {budget_error:.2e}")
            print(f"  Alpha range: [{min_alpha:.6f}, {max_alpha:.6f}]")
            print(f"  All alphas in [0,1]: {np.all((alpha_star >= -1e-8) & (alpha_star <= 1.0 + 1e-8))}")

        # Store prices for next batch (price smoothing)
        self.prev_prices = prices_star.copy()

        self.solution = {
            "status": "success",
            "alpha": alpha_star,
            "prices": prices_star,
            "aggregate_flow": aggregate_flow,
            "new_inventory": new_inventory,
            "welfare": welfare,
            "risk": risk,
            "objective_value": objective_value,
            "budget_value": budget_error,  # Should be ~0
            "pieces": self.pieces,
            "solver_info": {
                "method": getattr(result, "method", "unknown"),
                "iterations": getattr(result, "nit", 0),
                "function_evals": getattr(result, "nfev", 0),
                "constraint_violation": abs(budget_error)
            }
        }
        return self.solution

    def _diagnose_infeasibility(self, X: np.ndarray, v: np.ndarray, verbose: bool = False) -> str:
        """Diagnose why the auction might be infeasible"""
        reasons = []

        # Check if orders are one-sided
        aggregate_all = np.sum(X, axis=0)
        if verbose:
            print(f"Aggregate if all filled: {aggregate_all}")

        # Check for excessive imbalances
        for i, flow in enumerate(aggregate_all):
            if abs(flow) > 1e3:  # Large imbalance threshold
                token_name = ["USDC", "SOL", "ETH", "BTC"][i] if i < 4 else f"Token{i}"
                if flow > 0:
                    reasons.append(f"Excessive net buying in {token_name} ({flow:.1f})")
                else:
                    reasons.append(f"Excessive net selling in {token_name} ({flow:.1f})")

        # Check inventory bounds feasibility
        potential_inventory = self.market.inventory + aggregate_all
        q_min, q_max = self.market.inventory_bounds

        for i, (curr, min_val, max_val) in enumerate(zip(potential_inventory, q_min, q_max)):
            token_name = ["USDC", "SOL", "ETH", "BTC"][i] if i < 4 else f"Token{i}"
            if curr < min_val - 1e-6:
                reasons.append(f"{token_name} would hit lower bound ({curr:.1f} < {min_val:.1f})")
            if curr > max_val + 1e-6:
                reasons.append(f"{token_name} would hit upper bound ({curr:.1f} > {max_val:.1f})")

        # Check for degenerate orders (zero value with non-zero quantities)
        zero_value_orders = np.sum(np.abs(v) < 1e-12)
        if zero_value_orders > 0:
            reasons.append(f"{zero_value_orders} orders have zero/near-zero values")

        # Check budget constraint feasibility at extremes
        min_budget_if_all_filled = np.sum([np.dot(X[i], [1.0] * n_tokens) for i in range(len(X))])
        if abs(min_budget_if_all_filled) > 1e-6:
            reasons.append(f"Budget imbalance even with uniform prices: {min_budget_if_all_filled:.3f}")

        return "; ".join(reasons) if reasons else "No clear infeasibility detected"

    def print_solution(self):
        """Print detailed solution analysis"""
        if not self.solution or self.solution["status"] != "success":
            print("No valid solution to display")
            return

        sol = self.solution
        token_names = ["USDC", "SOL", "ETH", "BTC"][:self.market.num_tokens]

        print("\n" + "=" * 60)
        print("BATCH AUCTION CLEARING RESULTS")
        print("=" * 60)

        # Clearing prices
        print("\n📈 CLEARING PRICES:")
        for i, price in enumerate(sol["prices"]):
            symbol = " (numeraire)" if i == self.market.numeraire_idx else ""
            print(f"   {token_names[i]}: ${price:.6f}{symbol}")

        # Order fills
        print(f"\n📋 ORDER FILLS:")
        total_filled = 0
        total_volume = 0.0
        
        for i, (alpha, piece) in enumerate(zip(sol["alpha"], sol["pieces"])):
            if alpha > 1e-6:  # Only show meaningful fills
                print(f"   User {piece.user_id}, Piece {piece.piece_id}: {alpha:.1%} filled")
                filled_qty = alpha * piece.quantities
                piece_volume = 0.0
                
                for j, qty in enumerate(filled_qty):
                    if abs(qty) > 1e-6:
                        action = "receives" if qty > 0 else "pays"
                        print(f"     {action} {abs(qty):.6f} {token_names[j]}")
                        if j != self.market.numeraire_idx:
                            piece_volume += abs(qty) * sol["prices"][j]
                
                total_filled += 1
                total_volume += piece_volume

        print(f"   Total pieces filled: {total_filled}/{len(sol['pieces'])}")
        print(f"   Total trading volume: ${total_volume:.2f}")

        # Inventory changes
        print(f"\n📦 INVENTORY CHANGES:")
        for i in range(self.market.num_tokens):
            print(f"   {token_names[i]}: {self.market.inventory[i]:.3f} → {sol['new_inventory'][i]:.3f} (Δ{sol['aggregate_flow'][i]:.3f})")

        # Economic metrics
        print(f"\n💰 ECONOMIC METRICS:")
        print(f"   Total Welfare: ${sol['welfare']:.6f}")
        print(f"   Risk Cost (λ×R): ${self.market.risk_aversion * sol['risk']:.6f}")
        print(f"   Net Objective: ${sol['objective_value']:.6f}")
        print(f"   Budget Error: ${sol['budget_value']:.8f} (should be ≈ 0)")

        # Risk analysis - ENHANCED
        deviation = sol["new_inventory"] - self.market.target_inventory
        risk_contrib = self.market.covariance_matrix @ deviation
        print(f"\n⚠️  RISK ANALYSIS:")
        print(f"   Risk Aversion λ: {self.market.risk_aversion}")
        print(f"   Quadratic Risk R: {sol['risk']:.6f}")
        print(f"   Inventory Deviation: {deviation}")
        print(f"   Risk Gradient: {risk_contrib}")
        
        # Portfolio risk decomposition
        total_variance = deviation.T @ self.market.covariance_matrix @ deviation
        print(f"   Portfolio Variance: {total_variance:.6f}")

    def analyze_kkt_conditions(self):
        """CORRECTED KKT conditions analysis"""
        if not self.solution or self.solution["status"] != "success":
            print("No solution available for KKT analysis")
            return

        sol = self.solution
        X = np.array([piece.quantities for piece in self.pieces])
        v = np.array([piece.value for piece in self.pieces])

        print(f"\n🔍 KKT CONDITIONS ANALYSIS (CORRECTED)")
        print("=" * 50)

        # Compute Lagrange multipliers
        # For budget constraint (equality): μ can be any value
        # For inventory: λ_inv = λ * ∇R(q') = λ * Σ * (q' - q̄)
        deviation = sol["new_inventory"] - self.market.target_inventory
        lambda_inv = self.market.risk_aversion * (self.market.covariance_matrix @ deviation)

        # Estimate budget multiplier from stationarity condition
        # ∇_α L = -v + X * (p + λ_inv) = 0  => X * (p + λ_inv) = v
        # This is overdetermined, so we find least squares estimate
        try:
            combined_prices = sol["prices"] + lambda_inv
            # Solve: X @ mu_vec ≈ v  where mu_vec represents effective marginal costs
            mu_budget_vec = np.linalg.lstsq(X, v, rcond=None)[0]
            residual_norm = np.linalg.norm(X @ mu_budget_vec - v)
        except np.linalg.LinAlgError:
            mu_budget_vec = np.zeros(self.market.num_tokens)
            residual_norm = np.inf

        print(f"Estimated marginal costs: {mu_budget_vec}")
        print(f"Stationarity residual: {residual_norm:.6f}")

        # Budget constraint check
        budget_error = abs(sol["budget_value"])
        print(f"Budget constraint error: {budget_error:.8f}")

        # Detailed piece analysis
        kkt_violations = 0
        stationarity_errors = []

        print(f"\n📋 PIECE-BY-PIECE ANALYSIS:")
        for i, (alpha, piece) in enumerate(zip(sol["alpha"], sol["pieces"])):
            # Compute marginal value vs marginal cost
            marginal_value = piece.value
            
            # Marginal cost = p^T x_j + λ_inv^T x_j (inventory risk cost)
            price_cost = np.dot(sol["prices"], piece.quantities)
            inventory_cost = np.dot(lambda_inv, piece.quantities)
            total_marginal_cost = price_cost + inventory_cost

            # Stationarity condition check
            stationarity_error = marginal_value - total_marginal_cost
            stationarity_errors.append(abs(stationarity_error))

            # Determine fill status
            if alpha < 1e-8:
                fill_status = "Not filled"
            elif alpha > 0.99:
                fill_status = "Fully filled"
            else:
                fill_status = "Partially filled"

            print(f"\n  Piece {i} ({piece.user_id}) - {fill_status}:")
            print(f"    α = {alpha:.6f}")
            print(f"    Marginal value = {marginal_value:.6f}")
            print(f"    Price cost = {price_cost:.6f}")
            print(f"    Inventory cost = {inventory_cost:.6f}")
            print(f"    Total cost = {total_marginal_cost:.6f}")
            print(f"    Stationarity error = {stationarity_error:.6f}")

            # KKT complementary slackness check
            kkt_ok = True
            tolerance = 1e-4

            if 0.01 < alpha < 0.99:  # Interior solution
                if abs(stationarity_error) > tolerance:
                    print(f"    ❌ KKT violation: interior point but grad ≠ 0")
                    kkt_violations += 1
                    kkt_ok = False
            elif alpha < 0.01:  # At lower bound
                if stationarity_error < -tolerance:  # Gradient should be non-positive
                    print(f"    ❌ KKT violation: at lower bound but grad < 0")
                    kkt_violations += 1
                    kkt_ok = False
            elif alpha > 0.99:  # At upper bound
                if stationarity_error > tolerance:  # Gradient should be non-negative
                    print(f"    ❌ KKT violation: at upper bound but grad > 0")
                    kkt_violations += 1
                    kkt_ok = False

            if kkt_ok:
                print(f"    ✅ KKT conditions satisfied")

        # Summary
        print(f"\n📊 KKT SUMMARY:")
        print(f"  Total pieces: {len(sol['pieces'])}")
        print(f"  KKT violations: {kkt_violations}")
        print(f"  Mean stationarity error: {np.mean(stationarity_errors):.6f}")
        print(f"  Max stationarity error: {np.max(stationarity_errors):.6f}")
        print(f"  Solution quality: {'✅ OPTIMAL' if kkt_violations == 0 else '⚠️ SUBOPTIMAL'}")

        # Constraint qualification check
        self._check_constraint_qualification(X, sol)

    def _check_constraint_qualification(self, X: np.ndarray, sol: Dict):
        """Check Linear Independence Constraint Qualification (LICQ)"""
        print(f"\n🔍 CONSTRAINT QUALIFICATION CHECK:")
        
        alpha_star = sol["alpha"]
        
        # Find active constraints
        active_lower = np.where(alpha_star < 1e-8)[0]  # α_j = 0
        active_upper = np.where(alpha_star > 0.99)[0]   # α_j = 1
        
        print(f"  Active lower bounds (α=0): {len(active_lower)} pieces")
        print(f"  Active upper bounds (α=1): {len(active_upper)} pieces")
        
        # Budget constraint is always active (equality)
        # Inventory constraints may be active - check
        new_inv = sol["new_inventory"]
        q_min, q_max = self.market.inventory_bounds
        
        active_inv_lower = np.where(new_inv - q_min < 1e-6)[0]
        active_inv_upper = np.where(q_max - new_inv < 1e-6)[0]
        
        print(f"  Active inventory lower bounds: {len(active_inv_lower)} tokens")
        print(f"  Active inventory upper bounds: {len(active_inv_upper)} tokens")
        
        total_active = len(active_lower) + len(active_upper) + 1  # +1 for budget constraint
        total_vars = len(alpha_star) + self.market.num_tokens
        
        print(f"  Total active constraints: {total_active}")
        print(f"  Total variables: {total_vars}")
        
        if total_active <= total_vars:
            print(f"  ✅ LICQ potentially satisfied (active ≤ vars)")
        else:
            print(f"  ⚠️ Too many active constraints (over-constrained)")

    def _diagnose_infeasibility(self, X: np.ndarray, v: np.ndarray, verbose: bool = False) -> str:
        """Diagnose why the auction might be infeasible"""
        reasons = []

        # Check if orders are one-sided
        aggregate_all = np.sum(X, axis=0)
        if verbose:
            print(f"Aggregate if all filled: {aggregate_all}")

        # Check for excessive imbalances
        for i, flow in enumerate(aggregate_all):
            if abs(flow) > 1e3:  # Large imbalance threshold
                token_name = ["USDC", "SOL", "ETH", "BTC"][i] if i < 4 else f"Token{i}"
                if flow > 0:
                    reasons.append(f"Excessive net buying in {token_name} ({flow:.1f})")
                else:
                    reasons.append(f"Excessive net selling in {token_name} ({flow:.1f})")

        # Check inventory bounds feasibility
        potential_inventory = self.market.inventory + aggregate_all
        q_min, q_max = self.market.inventory_bounds

        for i, (curr, min_val, max_val) in enumerate(zip(potential_inventory, q_min, q_max)):
            token_name = ["USDC", "SOL", "ETH", "BTC"][i] if i < 4 else f"Token{i}"
            if curr < min_val - 1e-6:
                reasons.append(f"{token_name} would hit lower bound ({curr:.1f} < {min_val:.1f})")
            if curr > max_val + 1e-6:
                reasons.append(f"{token_name} would hit upper bound ({curr:.1f} > {max_val:.1f})")

        # Check for degenerate orders (zero value with non-zero quantities)
        zero_value_orders = np.sum(np.abs(v) < 1e-12)
        if zero_value_orders > 0:
            reasons.append(f"{zero_value_orders} orders have zero/near-zero values")

        # Check budget constraint feasibility - CORRECTED
        # For budget balance, we need the aggregate flow to be balanced at some price vector
        # If all orders are buy-only or sell-only in numeraire terms, it's impossible
        numeraire_flows = X[:, self.market.numeraire_idx]
        if np.all(numeraire_flows >= -1e-8):  # All receive numeraire (impossible to balance)
            reasons.append("All orders receive numeraire - no budget balance possible")
        elif np.all(numeraire_flows <= 1e-8):  # All pay numeraire
            reasons.append("All orders pay numeraire - no budget balance possible")

        return "; ".join(reasons) if reasons else "No clear infeasibility detected"

    def add_conditional_order(self, user_id: str, base_token: int, quote_token: int,
                              quantity: float, trigger_price: float, limit_price: float,
                              is_buy: bool = True):
        """Add a conditional order (stop-loss, take-profit). For now, treated as regular limit order."""
        self.add_limit_order(user_id, base_token, quantity, limit_price, is_buy)
        print(f"⚠️  Conditional order added as limit order (trigger at ${trigger_price})")

    def add_fee_structure(self, taker_fee_bps: float = 5.0, maker_rebate_bps: float = 1.0):
        """Add fee structure to market (stored only; not applied in objective)."""
        self.taker_fee_bps = taker_fee_bps
        self.maker_rebate_bps = maker_rebate_bps
        print(f"📊 Fee structure: {taker_fee_bps} bps taker fee, {maker_rebate_bps} bps maker rebate")

    def compute_transaction_costs(self, aggregate_flow: np.ndarray, penalty_factor: float = 0.01) -> float:
        """Compute L1 transaction cost penalty: γ||X||₁ (not included in objective by default)."""
        return penalty_factor * np.sum(np.abs(aggregate_flow))

    def get_market_depth(self, token_idx: int, price_levels: int = 5) -> Dict:
        """Analyze market depth for a specific token - ADDED"""
        if not self.solution or self.solution["status"] != "success":
            return {"error": "No solution available"}
        
        # Extract orders for the specific token
        buy_orders = []
        sell_orders = []
        
        for i, piece in enumerate(self.pieces):
            token_qty = piece.quantities[token_idx]
            numeraire_qty = piece.quantities[self.market.numeraire_idx]
            
            if abs(token_qty) > 1e-8 and abs(numeraire_qty) > 1e-8:
                implied_price = abs(numeraire_qty / token_qty)
                fill_rate = self.solution["alpha"][i]
                
                if token_qty > 0:  # Buy order
                    buy_orders.append((implied_price, abs(token_qty), fill_rate))
                else:  # Sell order
                    sell_orders.append((implied_price, abs(token_qty), fill_rate))
        
        # Sort by price
        buy_orders.sort(key=lambda x: x[0], reverse=True)  # Highest price first
        sell_orders.sort(key=lambda x: x[0])  # Lowest price first
        
        return {
            "clearing_price": self.solution["prices"][token_idx],
            "buy_orders": buy_orders[:price_levels],
            "sell_orders": sell_orders[:price_levels],
            "spread": sell_orders[0][0] - buy_orders[0][0] if buy_orders and sell_orders else np.nan
        }


def create_sample_market() -> Tuple[BatchAuctionEngine, MarketState]:
    """Create a sample 2-token market (USDC/SOL) for demonstration"""
    market = MarketState(
        num_tokens=2,
        numeraire_idx=0,  # USDC
        inventory=np.array([1000.0, -10.0]),  # Long USDC, short SOL
        target_inventory=np.array([0.0, 0.0]),  # Target balanced
        covariance_matrix=np.array([[1.0, 0.7], [0.7, 2.5]]),  # SOL more volatile, correlated
        risk_aversion=0.5,
        inventory_bounds=(
            np.array([-10000.0, -100.0]),  # Lower bounds
            np.array([10000.0, 100.0])     # Upper bounds
        )
    )
    engine = BatchAuctionEngine(market)
    return engine, market

def stress_test_market():
    """Stress test with edge cases - ENHANCED"""
    print("\n🧪 STRESS TESTING BATCH AUCTION")
    print("=" * 40)

    # Create market with extreme conditions
    market = MarketState(
        num_tokens=3,  # USDC, SOL, ETH
        numeraire_idx=0,
        inventory=np.array([100.0, -50.0, 20.0]),  # Imbalanced inventory
        target_inventory=np.array([0.0, 0.0, 0.0]),
        covariance_matrix=np.array(
            [
                [1.0, 0.8, 0.6],
                [0.8, 4.0, 2.5],  # High volatility and correlation
                [0.6, 2.5, 3.0]
            ]
        ),
        risk_aversion=2.0,  # High risk aversion
        inventory_bounds=(
            np.array([-1000.0, -200.0, -100.0]),
            np.array([1000.0, 200.0, 100.0])
        )
    )

    engine = BatchAuctionEngine(market)

    # Test Case 1: Balanced market (CORRECTED)
    print("\n📋 Test 1: Balanced market")
    engine.add_limit_order("Buyer1", 1, 10.0, 150.0, True)    # Buy SOL
    engine.add_limit_order("Seller1", 1, 8.0, 145.0, False)   # Sell SOL
    engine.add_limit_order("Buyer2", 2, 5.0, 3000.0, True)    # Buy ETH
    engine.add_limit_order("Seller2", 2, 4.0, 2950.0, False)  # Sell ETH

    result1 = engine.solve_batch_auction(verbose=False)
    print(f"Result: {result1['status']}")
    if result1['status'] == 'success':
        print(f"  SOL clearing price: ${result1['prices'][1]:.2f}")
        print(f"  ETH clearing price: ${result1['prices'][2]:.2f}")

    # Test Case 2: One-sided market
    print("\n📋 Test 2: One-sided market (corrected)")
    engine.pieces = []  # Clear orders
    
    # Add only buy orders (should fail or have zero fills)
    engine.add_limit_order("Buyer1", 1, 100.0, 150.0, True)  # Buy SOL
    engine.add_limit_order("Buyer2", 2, 50.0, 3000.0, True)  # Buy ETH

    result2 = engine.solve_batch_auction(verbose=False)
    print(f"Result: {result2['status']}")
    if result2["status"] == "failed":
        print(f"Reason: {result2.get('infeasible_reason', 'Unknown')}")

    # Test Case 3: Tight inventory constraints
    print("\n📋 Test 3: Tight inventory constraints")
    engine.pieces = []  # Clear orders
    
    # Set very tight inventory bounds
    engine.market.inventory_bounds = (
        np.array([90.0, -52.0, 18.0]),   # Very tight lower bounds
        np.array([110.0, -48.0, 22.0])   # Very tight upper bounds
    )

    # Add orders that would violate bounds
    engine.add_limit_order("BigSeller", 1, 200.0, 100.0, False)  # Massive sell
    engine.add_limit_order("BigBuyer", 1, 180.0, 120.0, True)    # Massive buy
    
    result3 = engine.solve_batch_auction(verbose=False)
    print(f"Result: {result3['status']}")
    if result3["status"] == "failed":
        print(f"Reason: {result3.get('infeasible_reason', 'Unknown')}")

def run_standard_demo():
    """Run the standard demo - ENHANCED"""
    print("\n🚀 Batch Auction Mathematical Engine")
    print("=====================================")

    # Create market
    engine, market = create_sample_market()

    print(f"\n📊 Market Configuration:")
    print(f"  Tokens: USDC (numéraire), SOL")
    print(f"  Current inventory: {market.inventory}")
    print(f"  Target inventory: {market.target_inventory}")
    print(f"  Risk aversion λ: {market.risk_aversion}")
    print(f"  Covariance matrix:")
    for i, row in enumerate(market.covariance_matrix):
        print(f"    {row}")

    # Add sample orders - BALANCED SET
    print(f"\n📝 Adding sample orders...")
    # Buy orders
    engine.add_limit_order("Alice", 1, 10.0, 130.0, is_buy=True)
    engine.add_limit_order("Charlie", 1, 5.0, 135.0, is_buy=True)
    
    # Sell orders
    engine.add_limit_order("Bob", 1, 15.0, 125.0, is_buy=False)
    engine.add_order_ladder("Dave", 1, [3.0, 4.0, 5.0], [128.0, 132.0, 136.0], is_buy=False)

    print(f"Total orders: {len(engine.pieces)}")

    # Solve and analyze
    result = engine.solve_batch_auction(verbose=True)
    if result["status"] == "success":
        engine.print_solution()
        engine.analyze_kkt_conditions()
        
        # Additional analysis
        depth = engine.get_market_depth(1)  # SOL depth
        if "error" not in depth:
            print(f"\n📊 MARKET DEPTH ANALYSIS (SOL):")
            print(f"  Clearing price: ${depth['clearing_price']:.6f}")
            print(f"  Bid-ask spread: ${depth.get('spread', 'N/A')}")
    else:
        print(f"❌ Failed: {result.get('message', 'Unknown error')}")
        if 'infeasible_reason' in result:
            print(f"   Reason: {result['infeasible_reason']}")

def custom_market_setup():
    """Allow user to create custom market - ENHANCED"""
    print("\n🔧 CUSTOM MARKET SETUP")
    print("=" * 30)

    try:
        # Get market parameters with validation
        num_tokens_input = input("Number of tokens (2-4): ") or "2"
        num_tokens = max(2, min(4, int(num_tokens_input)))

        risk_aversion_input = input("Risk aversion λ (0.1-5.0): ") or "1.0"
        risk_aversion = max(0.1, min(5.0, float(risk_aversion_input)))

        # Get initial inventory
        print(f"Enter initial inventory for {num_tokens} tokens:")
        inventory = []
        for i in range(num_tokens):
            token_name = ["USDC", "SOL", "ETH", "BTC"][i] if i < 4 else f"Token{i}"
            qty_input = input(f"  {token_name} inventory: ") or "0"
            inventory.append(float(qty_input))

        # Create market
        market = MarketState(
            num_tokens=num_tokens,
            risk_aversion=risk_aversion,
            inventory=np.array(inventory)
        )

        engine = BatchAuctionEngine(market)

        # Add orders interactively
        print(f"\n📝 Add orders (need both buys and sells for balance):")
        order_count = 0
        
        while True:
            print(f"\nCurrent orders: {len(engine.pieces)}")
            action = input("Add order? (y/n): ").strip().lower()
            if action != 'y':
                break

            try:
                user_id = input("User ID: ") or f"User{order_count}"
                
                print(f"Available tokens: {', '.join([f'{i}: {name}' for i, name in enumerate(['USDC', 'SOL', 'ETH', 'BTC'][:num_tokens])])}")
                token_idx = int(input(f"Token index (1-{num_tokens-1}, not 0=USDC): ") or "1")
                if token_idx == 0:
                    print("❌ Cannot directly trade numeraire token")
                    continue
                    
                quantity = float(input("Quantity: ") or "10.0")
                price = max(0.01, float(input("Price: ") or "100.0"))
                is_buy = input("Buy order? (y/n): ").strip().lower() == 'y'

                engine.add_limit_order(user_id, token_idx, quantity, price, is_buy)
                order_count += 1
                
            except ValueError as e:
                print(f"❌ Invalid input: {e}")
                continue

        # Solve custom market
        if engine.pieces:
            print(f"\n🔄 Solving auction...")
            result = engine.solve_batch_auction(verbose=True)
            if result["status"] == "success":
                engine.print_solution()
                engine.analyze_kkt_conditions()
            else:
                print(f"❌ Failed: {result}")
        else:
            print("No orders added. Exiting.")

    except (ValueError, KeyboardInterrupt) as e:
        print(f"\n❌ Setup cancelled or invalid input: {e}")

def multi_batch_simulation():
    """Simulate multiple auction batches over time - ENHANCED"""
    print("\n⏰ MULTI-BATCH SIMULATION")
    print("=" * 30)

    try:
        num_batches_input = input("Number of batches to simulate (3-10): ") or "5"
        num_batches = max(3, min(10, int(num_batches_input)))
        
        volatility_input = input("Market volatility (0.1-2.0): ") or "0.5"
        volatility = max(0.1, min(2.0, float(volatility_input)))
        
    except ValueError:
        num_batches = 5
        volatility = 0.5
        print(f"Using defaults: {num_batches} batches, {volatility} volatility")

    # Initialize market
    engine, market = create_sample_market()

    # Track metrics over time
    price_history = []
    inventory_history = []
    welfare_history = []
    risk_history = []

    base_price = 128.0  # Starting SOL price

    for batch_num in range(num_batches):
        print(f"\n🔄 BATCH {batch_num + 1}/{num_batches}")
        print("-" * 25)

        # Clear previous orders
        engine.pieces = []

        try:
            # Generate random but balanced orders for this batch
            rng = np.random.default_rng(seed=42 + batch_num)  # Reproducible
            
            num_orders = rng.integers(4, 10)  # 4-9 orders per batch
            buy_orders = 0
            sell_orders = 0

            for i in range(num_orders):
                user_id = f"B{batch_num}U{i}"
                
                # Ensure we have both buys and sells
                if i < num_orders // 2:
                    is_buy = True
                    buy_orders += 1
                else:
                    is_buy = False
                    sell_orders += 1
                
                quantity = float(rng.uniform(5.0, 25.0))

                # Price varies around current market level with volatility
                price_shock = float(rng.normal(0, volatility * base_price * 0.05))
                if is_buy:
                    price = max(50.0, base_price - abs(rng.normal(2.0, 1.0)) + price_shock)
                else:
                    price = base_price + abs(rng.normal(2.0, 1.0)) + price_shock

                engine.add_limit_order(user_id, 1, quantity, price, is_buy)

            print(f"  Generated {buy_orders} buy orders, {sell_orders} sell orders")

        except Exception as e:
            print(f"  ❌ Error generating orders: {e}")
            continue

        # Solve batch
        result = engine.solve_batch_auction(verbose=False)

        if result["status"] == "success":
            # Record metrics
            price_history.append(result["prices"].copy())
            inventory_history.append(result["new_inventory"].copy())
            welfare_history.append(float(result["welfare"]))
            risk_history.append(float(result["risk"]))

            # Update market state for next batch
            engine.market.inventory = result["new_inventory"]
            
            # Update base price for next batch (with momentum)
            new_price = result["prices"][1]
            base_price = 0.7 * base_price + 0.3 * new_price  # Price momentum

            print(f"  ✅ Cleared {np.sum(result['alpha'] > 0.01)}/{len(result['pieces'])} orders")
            print(f"  SOL price: ${result['prices'][1]:.2f}")
            print(f"  Welfare: ${result['welfare']:.2f}")
            print(f"  Risk: {result['risk']:.4f}")
            print(f"  Budget error: {abs(result['budget_value']):.2e}")
            
        else:
            print(f"  ❌ Batch failed: {result.get('message', 'Unknown error')}")
            # Reset to try to continue
            if 'infeasible' in result.get('message', '').lower():
                print("  🔄 Attempting to continue with reduced position...")
                engine.market.inventory *= 0.8  # Reduce positions
                continue
            else:
                break

    # Summary statistics - ENHANCED
    if price_history:
        print(f"\n📊 SIMULATION SUMMARY:")
        print("=" * 30)

        prices = np.array(price_history)
        inventories = np.array(inventory_history)
        
        if len(prices) > 1:
            # Price statistics
            sol_prices = prices[:, 1]
            price_returns = np.diff(np.log(sol_prices)) if len(sol_prices) > 1 else np.array([])
            
            print(f"SOL Price Evolution:")
            print(f"  Initial: ${sol_prices[0]:.2f}")
            print(f"  Final: ${sol_prices[-1]:.2f}")
            print(f"  Average: ${np.mean(sol_prices):.2f}")
            print(f"  Volatility (price std): ${np.std(sol_prices):.2f}")
            if len(price_returns) > 0:
                print(f"  Return volatility: {np.std(price_returns) * 100:.2f}%")

            # Inventory evolution
            print(f"\nInventory Evolution:")
            print(f"  USDC: {inventories[0][0]:.1f} → {inventories[-1][0]:.1f}")
            print(f"  SOL: {inventories[0][1]:.1f} → {inventories[-1][1]:.1f}")
            if inventories.shape[1] > 2:
                print(f"  ETH: {inventories[0][2]:.1f} → {inventories[-1][2]:.1f}")

            # Performance metrics
            print(f"\nPerformance Metrics:")
            print(f"  Total Welfare: ${sum(welfare_history):.2f}")
            print(f"  Average Risk per batch: {np.mean(risk_history):.4f}")
            print(f"  Risk trend: {'📈 Increasing' if risk_history[-1] > risk_history[0] else '📉 Decreasing'}")
            
            # Market efficiency
            successful_batches = len(price_history)
            efficiency = successful_batches / num_batches * 100
            print(f"  Market efficiency: {efficiency:.1f}% ({successful_batches}/{num_batches} successful)")

def interactive_mode():
    """Enhanced interactive terminal mode"""
    print("\n🚀 Advanced Batch Auction Mathematical Engine")
    print("=============================================")
    print("Mathematical Framework: Quadratic Programming with Risk Management")

    # Menu system
    while True:
        print(f"\n📋 MENU:")
        print("1. Run standard demo (balanced market)")
        print("2. Run stress tests (edge cases)")
        print("3. Custom market setup")
        print("4. Multi-batch simulation")
        print("5. Mathematical framework info")
        print("6. Exit")

        try:
            choice = input("\nSelect option (1-6): ").strip()

            if choice == "1":
                run_standard_demo()
            elif choice == "2":
                stress_test_market()
            elif choice == "3":
                custom_market_setup()
            elif choice == "4":
                multi_batch_simulation()
            elif choice == "5":
                print_mathematical_framework()
            elif choice == "6":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid option")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def print_mathematical_framework():
    """Print detailed mathematical framework explanation"""
    print(f"\n📖 MATHEMATICAL FRAMEWORK")
    print("=" * 40)
    
    print("""
🔢 OPTIMIZATION PROBLEM:
   max_{α,p} Σ αⱼ vⱼ - λ/2 (q' - q̄)ᵀ Σ (q' - q̄) - η||p - p_prev||²

🎯 CONSTRAINTS:
   • 0 ≤ αⱼ ≤ 1                    (fill rates)
   • pᵀ(XᵀΑ) = 0                   (budget balance)
   • q_min ≤ q' ≤ q_max           (inventory bounds)
   • p_numeraire = 1              (price normalization)

🔧 VARIABLES:
   • α ∈ [0,1]ⁿ: fill rates for n order pieces
   • p ∈ ℝ₊ᵈ: clearing prices for d tokens
   • q' = q + XᵀΑ: new inventory after clearing

📊 COMPONENTS:
   • X ∈ ℝⁿˣᵈ: order quantity matrix
   • v ∈ ℝⁿ: order values
   • Σ ∈ ℝᵈˣᵈ: covariance matrix (PSD)
   • λ > 0: risk aversion parameter

✅ KKT CONDITIONS:
   • Stationarity: ∇_α L = -v + Xμ = 0
   • Primal feasibility: all constraints satisfied
   • Dual feasibility: multipliers have correct signs
   • Complementary slackness: αⱼ(1-αⱼ) = 0 or gradient conditions
    """)

def main():
    """Enhanced main function with comprehensive testing"""
    parser = argparse.ArgumentParser(description="Advanced Batch Auction Mathematical Engine")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--test", choices=["standard", "stress", "custom", "multi"],
                        help="Run specific test suite")
    parser.add_argument("--config", help="JSON config file path")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    try:
        if args.test:
            if args.test == "standard":
                run_standard_demo()
            elif args.test == "stress":
                stress_test_market()
            elif args.test == "custom":
                custom_market_setup()
            elif args.test == "multi":
                multi_batch_simulation()
        elif args.interactive:
            interactive_mode()
        elif args.config:
            try:
                with open(args.config, 'r') as f:
                    config = json.load(f)
                print(f"Running with config: {args.config}")
                # TODO: Implement config-based market creation
                print("⚠️ Config-based setup not implemented yet")
            except FileNotFoundError:
                print(f"❌ Config file not found: {args.config}")
                sys.exit(1)
            except json.JSONDecodeError as e:
                print(f"❌ Invalid JSON in config file: {e}")
                sys.exit(1)
        else:
            # Default: run comprehensive demo
            print("🚀 COMPREHENSIVE BATCH AUCTION DEMO")
            print("=" * 40)
            print("Running standard demo then stress tests...\n")

            run_standard_demo()
            stress_test_market()

            print("\n✨ All tests completed!")
            print("Use --interactive for manual testing or --test <type> for specific tests")

    except KeyboardInterrupt:
        print("\n👋 Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()