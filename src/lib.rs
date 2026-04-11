use wasm_bindgen::prelude::*;

/// Ported from csaps: calculates normalized smoothing parameter
fn normalize_smooth(x: &[f64], w: &[f64], smooth: f64) -> f64 {
    let span = x.last().unwrap() - x.first().unwrap();
    let dx_sq_sum: f64 = x.windows(2).map(|win| (win[1] - win[0]).powi(2)).sum();
    let eff_x = 1.0 + span.powi(2) / dx_sq_sum;

    let w_sum: f64 = w.iter().sum();
    let w_sq_sum: f64 = w.iter().map(|&v| v.powi(2)).sum();
    let eff_w = w_sum.powi(2) / w_sq_sum;

    let k = 80.0 * span.powi(3) * (x.len() as f64).powi(-2) * eff_x.powf(-0.5) * eff_w.powf(-0.5);
    smooth / (smooth + (1.0 - smooth) * k)
}

#[wasm_bindgen]
pub fn solve_spline(
    x: &[f64],
    y: &[f64],
    w: &[f64],
    smooth: f64,
    use_normalized: bool
) -> Vec<f64> {
    let n = x.len();
    if n < 3 { return vec![]; } // Guard for simplistic cases

    let p_smooth = if use_normalized { normalize_smooth(x, w, smooth) } else { smooth };
    let pp = 6.0 * (1.0 - p_smooth);

    let dx: Vec<f64> = x.windows(2).map(|win| win[1] - win[0]).collect();
    let dy: Vec<f64> = y.windows(2).map(|win| win[1] - win[0]).collect();
    let dy_dx: Vec<f64> = dy.iter().zip(dx.iter()).map(|(dy, dx)| dy / dx).collect();

    let m = n - 2; // Size of the linear system

    // 1. Construct Q rows conceptually
    let mut q_rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
    for i in 0..m {
        let r0 = 1.0 / dx[i];
        let r1 = 1.0 / dx[i + 1];
        q_rows[i].push((i, r0));
        q_rows[i].push((i + 1, -(r0 + r1)));
        q_rows[i].push((i + 2, r1));
    }

    // 2. Build Pentadiagonal Matrix Bands
    let mut diag = vec![0.0; m];
    let mut off1 = vec![0.0; m - 1];
    let mut off2 = vec![0.0; m - 2];

    for i in 0..m {
        // Add p * R
        diag[i] += p_smooth * 2.0 * (dx[i] + dx[i + 1]);
        if i < m - 1 { off1[i] += p_smooth * dx[i + 1]; }

        // Add pp * Q W Q^T
        for j in i..usize::min(i + 3, m) {
            let mut dot = 0.0;
            for &(c_i, v_i) in &q_rows[i] {
                for &(c_j, v_j) in &q_rows[j] {
                    if c_i == c_j { dot += (v_i * v_j) / w[c_i]; }
                }
            }
            if j == i { diag[i] += pp * dot; }
            else if j == i + 1 { off1[i] += pp * dot; }
            else if j == i + 2 { off2[i] += pp * dot; }
        }
    }

    // 3. RHS vector (b)
    let mut b = vec![0.0; m];
    for i in 0..m { b[i] = dy_dx[i + 1] - dy_dx[i]; }

    // 4. In-Place Symmetric Pentadiagonal Factorization & Solve (O(N) Direct Solver)
    for i in 0..m - 2 {
        let m1 = off1[i] / diag[i];
        diag[i + 1] -= m1 * off1[i];
        off1[i + 1] -= m1 * off2[i];
        b[i + 1] -= m1 * b[i];

        let m2 = off2[i] / diag[i];
        diag[i + 2] -= m2 * off2[i];
        b[i + 2] -= m2 * b[i];
    }

    if m > 1 {
        let i = m - 2;
        let m1 = off1[i] / diag[i];
        diag[i + 1] -= m1 * off1[i];
        b[i + 1] -= m1 * b[i];
    }

    let mut u = vec![0.0; m];
    u[m - 1] = b[m - 1] / diag[m - 1];
    if m > 1 {
        u[m - 2] = (b[m - 2] - off1[m - 2] * u[m - 1]) / diag[m - 2];
    }
    for i in (0..m - 2).rev() {
        u[i] = (b[i] - off1[i] * u[i + 1] - off2[i] * u[i + 2]) / diag[i];
    }

    // 5. Reconstruct Spline Coefficients (Matching PPoly)
    let mut u_pad = vec![0.0; n];
    for i in 0..m { u_pad[i + 1] = u[i]; }

    let mut d1_pad = vec![0.0; n + 1];
    for i in 0..n - 1 { d1_pad[i + 1] = (u_pad[i + 1] - u_pad[i]) / dx[i]; }

    let mut d2 = vec![0.0; n];
    for i in 0..n { d2[i] = d1_pad[i + 1] - d1_pad[i]; }

    let mut yi = vec![0.0; n];
    let mut pu = vec![0.0; n];
    for i in 0..n {
        yi[i] = y[i] - pp * (1.0 / w[i]) * d2[i];
        pu[i] = p_smooth * u_pad[i];
    }

    // Flat array to map directly to a JS Float64Array
    let mut coeffs = vec![0.0; 4 * (n - 1)];
    for i in 0..n - 1 {
        coeffs[i] = (pu[i + 1] - pu[i]) / dx[i]; // c1
        coeffs[(n - 1) + i] = 3.0 * pu[i];       // c2
        coeffs[2 * (n - 1) + i] = (yi[i + 1] - yi[i]) / dx[i] - dx[i] * (2.0 * pu[i] + pu[i + 1]); // c3
        coeffs[3 * (n - 1) + i] = yi[i];         // c4
    }

    coeffs // Automatically bound to JS as a Float64Array
}