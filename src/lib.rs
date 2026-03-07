use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmSpline {
    coeffs: Vec<f64>,
}

#[wasm_bindgen]
impl WasmSpline {
    #[wasm_bindgen(constructor)]
    pub fn new(x: &[f64], y: &[f64], p: f64) -> WasmSpline {
        let n = x.len();
        // Pre-calculate h (intervals)
        let mut h = vec![0.0; n - 1];
        for i in 0..n - 1 {
            h[i] = x[i + 1] - x[i];
        }

        let mut r_diag = vec![0.0; n - 2];
        let mut r_off = vec![0.0; n - 3];
        let mut rhs = vec![0.0; n - 2];

        for i in 0..n - 2 {
            r_diag[i] = (h[i] + h[i + 1]) / 3.0;
            if i < n - 3 {
                r_off[i] = h[i + 1] / 6.0;
            }
            rhs[i] = (y[i + 2] - y[i + 1]) / h[i + 1] - (y[i + 1] - y[i]) / h[i];
        }

        let mu = (1.0 - p) / p;

        // Thomas Algorithm
        let n_rhs = rhs.len();
        let mut cp = vec![0.0; n_rhs];
        let mut dp = vec![0.0; n_rhs];
        let mut u = vec![0.0; n_rhs];

        let d0 = r_diag[0] + mu * (1.0 / h[0] + 1.0 / h[1]);
        cp[0] = r_off[0] / d0;
        dp[0] = rhs[0] / d0;

        for i in 1..n_rhs {
            let penalty = 1.0 / h[i] + if i < n_rhs - 1 { 1.0 / h[i + 1] } else { 0.0 };
            let di = r_diag[i] + mu * penalty;
            let m = di - r_off[i - 1] * cp[i - 1];
            if i < n_rhs - 1 {
                cp[i] = r_off[i] / m;
            }
            dp[i] = (rhs[i] - r_off[i - 1] * dp[i - 1]) / m;
        }

        u[n_rhs - 1] = dp[n_rhs - 1];
        for i in (0..n_rhs - 1).rev() {
            u[i] = dp[i] - cp[i] * u[i + 1];
        }

        // Expand u with boundary conditions (0 at ends)
        let mut full_u = vec![0.0; n];
        for i in 0..n - 2 {
            full_u[i + 1] = u[i];
        }

        // Compute 4 coefficients for each of the (n-1) segments
        let mut coeffs = vec![0.0; (n - 1) * 4];
        for i in 0..n - 1 {
            let idx = i * 4;
            let hi = h[i];
            coeffs[idx] = (full_u[i + 1] - full_u[i]) / (6.0 * hi);
            coeffs[idx + 1] = full_u[i] / 2.0;
            coeffs[idx + 2] = (y[i + 1] - y[i]) / hi - (hi * (2.0 * full_u[i] + full_u[i + 1])) / 6.0;
            coeffs[idx + 3] = y[i];
            let go = vec![0.0, 3];
        }

        WasmSpline { coeffs }
    }

    // Access the pointer to the coefficients in WASM Memory
    pub fn get_coeffs_ptr(&self) -> *const f64 {
        self.coeffs.as_ptr()
    }
}