import numpy as np
import time
from scipy.stats import norm
from scipy.integrate import quad

'''PARAMETRI'''
S = 105              
K = 105                 
T = 1                   
r = 0.05                
sigma = 0.2           
N = 1000 


v0 = 0.04       # varianza iniziale
kappa = 1.5     # velocità di mean reversion
theta = 0.04    # varianza di lungo periodo
sigma_v = 0.3   # vol-of-vol
rho = -0.7      # correlazione


alpha = 1.5   # damping (>0)
q = 0.0       # dividend yield
N = 4096      # punti FFT (tipicamente potenza di 2)
eta = 0.25    # passo in frequenza



'''Black-Scholes'''

def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

   
    
start_bs = time.time()
call_price_bs = black_scholes(S, K, T, r, sigma, option_type="call")
end_bs = time.time()



'''Heston (integrale diretto)'''

def heston_price(S, K, T, r, v0, kappa, theta, sigma_v, rho, option_type="call", q=0.0,
                 int_upper=150.0):
    
    i = 1j
    a = kappa * theta
    x0 = np.log(S)

    def char_func(phi, Pnum):
        # u e b secondo Heston; Little Trap per stabilità
        u = 0.5 if Pnum == 1 else -0.5 # P_1 e P_2
        b = kappa - rho * sigma_v if Pnum == 1 else kappa

        # d e g (nota: sqrt complessa)
        d = np.sqrt((rho*sigma_v*i*phi - b)**2 - (sigma_v**2)*(2*u*i*phi - phi**2))
        gp = (b - rho*sigma_v*i*phi + d) / (b - rho*sigma_v*i*phi - d)

        # Evita |gp| ~ 1
        # Formulazione 'trap': usa gp così che |gp| < 1
        exp_dT = np.exp(d * T)
        one_minus_gp = 1.0 - gp
        one_minus_gp_exp = 1.0 - gp * exp_dT

        # piccole salvaguardie numeriche
        eps = 1e-14
        one_minus_gp = np.where(np.abs(one_minus_gp) < eps, eps, one_minus_gp)
        one_minus_gp_exp = np.where(np.abs(one_minus_gp_exp) < eps, eps, one_minus_gp_exp)

        C = (r - q) * i * phi * T + (a / (sigma_v**2)) * (
            (b - rho*sigma_v*i*phi + d) * T - 2.0 * np.log(one_minus_gp_exp / one_minus_gp)
        )
        D = ((b - rho*sigma_v*i*phi + d) / (sigma_v**2)) * (
            (1.0 - exp_dT) / one_minus_gp_exp
        )
        return np.exp(C + D * v0 + i * phi * x0)

    def Pj(j):
        # integrando reale; gestiamo la singolarità in 0 implicitamente
        def integrand(phi):
            return np.real(np.exp(-i*phi*np.log(K)) * char_func(phi, j) / (i*phi))
        # integriamo da 0 a un limite alto (tipicamente 100–200)
        val, _ = quad(integrand, 0.0, int_upper, limit=500, epsabs=1e-9, epsrel=1e-7)
        return 0.5 + val/np.pi

    P1 = Pj(1)
    P2 = Pj(2)

    call = S*np.exp(-q*T)*P1 - K*np.exp(-r*T)*P2
    if option_type.lower() == "call":
        return float(call)
    else:
        # Put da parità
        return float(call - S*np.exp(-q*T) + K*np.exp(-r*T))



start_heston = time.time()
call_price_heston = heston_price(S, K, T, r, v0, kappa, theta, sigma_v, rho, option_type="call")
end_heston = time.time()


'''Heston con Carr-Madan'''


def heston_price_fft(S, K, T, r, v0, kappa, theta, sigma_v, rho,
                     option_type="call", q=0.0,
                     alpha=1.5, N=4096, eta=0.05):
    i = 1j

    # Caratteristica risk-neutral di ln(S_T)
    def phi(u):
        a = kappa * theta
        b_h = kappa
        d = np.sqrt((rho*sigma_v*i*u - b_h)**2 + (sigma_v**2)*(i*u + u**2))
        g = (b_h - rho*sigma_v*i*u + d) / (b_h - rho*sigma_v*i*u - d)

        C = (r - q)*i*u*T + (a/(sigma_v**2)) * (
            (b_h - rho*sigma_v*i*u + d)*T - 2*np.log((1 - g*np.exp(d*T))/(1 - g))
        )
        D = ((b_h - rho*sigma_v*i*u + d)/(sigma_v**2)) * (
            (1 - np.exp(d*T))/(1 - g*np.exp(d*T))
        )
        return np.exp(C + D*v0 + i*u*np.log(S))

    # Griglia in frequenza
    u = np.arange(N) * eta

    # Pesi trapezoidali
    w = np.ones(N)
    w[0] = 0.5
    w[-1] = 0.5
    w *= eta

    # psi(u) di Carr-Madan
    uj = u - i*(alpha + 1.0)
    numerator = np.exp(-r*T) * phi(uj)
    denominator = (alpha**2 + alpha - u**2) + i*(2*alpha + 1.0)*u
    denominator = np.where(np.abs(denominator) < 1e-14, 1e-14, denominator)  # guardia numerica
    psi = numerator / denominator

    # Spaziatura in log-strike
    lam = 2.0*np.pi / (N * eta)                 # passo in k
    k_min = np.log(S) - (N * lam) / 2.0         # shift (inizio griglia)
    k_grid = k_min + np.arange(N) * lam         # griglia di log-strike

    # Input FFT 
    fft_input = np.exp(-1j * k_min * u) * psi * w
    fft_vals = np.fft.fft(fft_input)
    fft_real = np.real(fft_vals)

    # Prezzi delle call su tutta la griglia k
    C_k = (np.exp(-alpha * k_grid) / np.pi) * fft_real

    # Interpola al log-strike desiderato
    k_target = np.log(K)
    if k_target <= k_grid[0]:
        price_call = C_k[0]
    elif k_target >= k_grid[-1]:
        price_call = C_k[-1]
    else:
        idx = np.searchsorted(k_grid, k_target) - 1
        t = (k_target - k_grid[idx]) / (k_grid[idx+1] - k_grid[idx])
        price_call = C_k[idx] * (1 - t) + C_k[idx+1] * t

    if option_type.lower() == "call":
        return float(price_call)
    else:
        # put by parity
        return float(price_call - S*np.exp(-q*T) + K*np.exp(-r*T))



start_heston_fft = time.time()
call_price_heston_fft = heston_price_fft(S, K, T, r, v0, kappa, theta, sigma_v, rho, option_type="call")
end_heston_fft = time.time()




print(f"Black-Scholes Call Price: {call_price_bs:.4f} (tempo: {end_bs-start_bs:.6f} s)")
print(f"Heston Call Price: {call_price_heston:.4f} (tempo: {end_heston-start_heston:.6f} s)")
print(f"Heston con FFT Call Price: {call_price_heston_fft:.4f} (tempo: {end_heston_fft-start_heston_fft:.6f} s)")