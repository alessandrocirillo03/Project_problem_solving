import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize, newton

class YieldCurveEngine:
    """
    Constructs robust zero-coupon spot rate curves and calibrates 
    the Nelson-Siegel-Svensson parametric model.
    
    This engine establishes the "term structure of spot rates", which is 
    an increasing collection of rates defined as i(t,t+k).
    """
    def __init__(self, maturities, par_yields):
        self.maturities = np.array(maturities)
        self.par_yields = np.array(par_yields)
        
        self.spot_rates = self._bootstrap_curve()
        self.nss_params = self._calibrate_nss()

    def _bootstrap_curve(self):
        """
        Derives spot rates from empirical par yields via cubic spline interpolation 
        and iterative cash flow discounting algorithms.
        
        According to the theorems of Arbitrage, the price of a coupon bond 
        must equal the sum of its discounted cash flows[cite: 324, 325]. This 
        method reverses that linear price property to isolate the spot rates.
        """
        max_mat = int(np.max(self.maturities))
        nodes = np.arange(0.5, max_mat + 0.5, 0.5)
        
        cs = CubicSpline(self.maturities, self.par_yields)
        interp_par = cs(nodes)
        
        spot_rates = np.zeros(len(nodes))
        
        for i, t in enumerate(nodes):
            if t <= 1.0:
                spot_rates[i] = interp_par[i]
            else:
                cpn = interp_par[i] / 2
                pv_coupons = sum([cpn / (1 + spot_rates[j]/2)**(2*nodes[j]) for j in range(i)])
                
                # Algebraically solving for the final zero rate at maturity node T.
                # This ensures the maturity condition and positivity property are maintained.
                spot_rates[i] = 2 * (((100 + cpn) / (100 - pv_coupons))**(1 / (2*t)) - 1)
                
        return pd.Series(spot_rates, index=nodes)

    def _nss_formula(self, t, beta0, beta1, beta2, beta3, tau1, tau2):
        """Mathematical representation of the NSS forward-rate decay model."""
        term1 = (1 - np.exp(-t / tau1)) / (t / tau1)
        term2 = term1 - np.exp(-t / tau1)
        term3 = ((1 - np.exp(-t / tau2)) / (t / tau2)) - np.exp(-t / tau2)
        
        return beta0 + beta1 * term1 + beta2 * term2 + beta3 * term3

    def _nss_error(self, params):
        """Objective function calculating Sum of Squared Errors (SSE)."""
        beta0, beta1, beta2, beta3, tau1, tau2 = params
        model_yields = self._nss_formula(self.spot_rates.index.values, 
                                         beta0, beta1, beta2, beta3, tau1, tau2)
        return np.sum((self.spot_rates.values - model_yields)**2)

    def _calibrate_nss(self):
        """Optimizes non-convex NSS parameters using L-BFGS-B bounded minimization."""
        initial_guess = [0.03, -0.02, 0.01, 0.01, 1.5, 5.0]
        bounds = ((0, 0.15), (-0.15, 0.15), (-0.15, 0.15), (-0.15, 0.15), (0.1, 10), (0.1, 20))
        
        result = minimize(self._nss_error, initial_guess, method='L-BFGS-B', bounds=bounds)
        return result.x

    def get_spot_rate(self, t):
        """Returns the continuous, smoothed NSS zero rate for any precise maturity t."""
        return self._nss_formula(t, *self.nss_params)


class BondAnalytics:
    """
    Computes theoretical pricing, Z-spreads, Effective Duration, and Effective Convexity.
    """
    def __init__(self, face_value, coupon_rate, maturity_years, yield_curve):
        # A coupon bond is characterized by its nominal value, coupon rate, and periodicity[cite: 319, 320, 321].
        self.face = face_value
        self.coupon = coupon_rate
        self.maturity = maturity_years
        self.curve = yield_curve
        self.cash_flows = self._generate_cash_flows()

    def _generate_cash_flows(self):
        """Generates rigorous semi-annual cash flows and exact timing nodes."""
        periods = np.arange(0.5, self.maturity + 0.5, 0.5)
        cfs = np.full(len(periods), self.face * (self.coupon / 2))
        cfs[-1] += self.face 
        return pd.Series(cfs, index=periods)

    def price_from_spread(self, z_spread_bps, curve_shift_bps=0):
        """
        Prices the bond by discounting cash flows at the baseline NSS spot rate 
        plus the Z-spread premium. 
        
        This aligns with the AOA price of a coupon bond[cite: 324, 325], while
        the added spread accounts for effects like credit risk or liquidity that
        cause underpricing relative to the risk-free curve[cite: 636, 637].
        """
        spread_decimal = z_spread_bps / 10000
        shift_decimal = curve_shift_bps / 10000
        pv = 0
        
        for t, cf in self.cash_flows.items():
            base_rate = self.curve.get_spot_rate(t)
            adjusted_rate = base_rate + spread_decimal + shift_decimal
            pv += cf / (1 + adjusted_rate / 2)**(2 * t)
            
        return pv

    def calculate_z_spread(self, market_price):
        """
        Solves for the exact Z-spread that equilibrates the model price to the 
        empirical market price.
        """
        def objective(spread):
            return self.price_from_spread(spread) - market_price
        
        z_spread = newton(objective, x0=100)
        return z_spread

    def calculate_risk_metrics(self, market_price, shock_bps=30):
        """
        Computes Effective Duration and Effective Convexity via programmatic curve shifting.
        
        These metrics are used to hedge the risk of interest rate variation through 
        immunization. This programmatic approach approximates the Taylor 
        expansion components used for infinitesimal parallel shifts of the rate curve 
        [cite: 484, 488, 489].
        """
        z_spread = self.calculate_z_spread(market_price)
        p_0 = market_price
        
        # Dynamically re-price the bond under symmetric shocked curve scenarios
        p_down = self.price_from_spread(z_spread, curve_shift_bps=-shock_bps)
        p_up = self.price_from_spread(z_spread, curve_shift_bps=shock_bps)
        
        shock_decimal = shock_bps / 10000
        
        # Approximates Duration (first derivative sensitivity) [cite: 423, 424]
        eff_dur = (p_down - p_up) / (2 * p_0 * shock_decimal)
        
        # Approximates Convexity (second derivative sensitivity) [cite: 461, 462, 463]
        eff_conv = (p_down + p_up - 2 * p_0) / (p_0 * shock_decimal**2)
        
        return {
            "Z-Spread (bps)": z_spread, 
            "Effective Duration": eff_dur, 
            "Effective Convexity": eff_conv
        }

if __name__ == "__main__":
    # 1. Market Data: U.S. Treasury Par Yields (April 2026)
    treasury_maturities = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 30.0]
    treasury_par_yields = [0.0510, 0.0394, 0.0390, 0.0398, 0.0426, 0.0431, 0.0488] 
    
    print("--- Initializing Yield Curve Engine ---")
    engine = YieldCurveEngine(treasury_maturities, treasury_par_yields)
    print("Calibrated NSS Parameters (beta0, beta1, beta2, beta3, tau1, tau2):")
    print(np.round(engine.nss_params, 4))
    
    # 2. Corporate Bond Data: Apple Inc. 1.7% maturing Aug 2031
    face_value = 100.0
    coupon_rate = 0.017  
    maturity_years = 5.5 
    market_price = 88.36
    
    print("\n--- Analyzing Corporate Bond (AAPL 1.7% 2031) ---")
    aapl_bond = BondAnalytics(face_value, coupon_rate, maturity_years, engine)
    
    # 3. Calculate Spread and Risk Metrics
    risk_metrics = aapl_bond.calculate_risk_metrics(market_price, shock_bps=30)
    
    print(f"Observed Market Price: ${market_price}")
    print(f"Calculated Z-Spread: {risk_metrics['Z-Spread (bps)']:.2f} bps")
    print(f"Effective Duration: {risk_metrics['Effective Duration']:.2f}")
    print(f"Effective Convexity: {risk_metrics['Effective Convexity']:.2f}")

    # 4. Generate Visualization
    print("\n--- Generating Yield Curve Visualization ---")
    t_plot = np.linspace(0.5, 30, 100)
    nss_yields = [engine.get_spot_rate(t) * 100 for t in t_plot] 

    plt.figure(figsize=(10, 6))
    plt.plot(t_plot, nss_yields, label='NSS Calibrated Spot Curve', color='blue', linewidth=2)
    plt.scatter(treasury_maturities, [y * 100 for y in treasury_par_yields], 
                color='red', label='Observed Par Yields', zorder=5)

    plt.title('US Treasury Yield Curve Calibration (April 2026 Data)', fontsize=14)
    plt.xlabel('Maturity (Years)', fontsize=12)
    plt.ylabel('Yield (%)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
