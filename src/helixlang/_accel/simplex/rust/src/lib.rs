use pyo3::prelude::*;

const INF: f64 = f64::INFINITY;

fn simplex_run(tab: &mut [Vec<f64>], basis: &mut [i64], obj: &[f64],
               n_vars: usize, forbidden: &[i64], eps: f64,
               max_iter: i64) -> &'static str {
    let n_rows = tab.len();
    let rhs_col = n_vars;
    let mut iters: i64 = 0;
    loop {
        // reduced costs, smallest-index entering (skip basis + forbidden)
        let mut entering: i64 = -1;
        let mut cb: Vec<f64> = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            cb.push(obj[basis[i] as usize]);
        }
        for j in 0..n_vars {
            if basis.contains(&(j as i64)) { continue; }
            if forbidden.binary_search(&(j as i64)).is_ok() { continue; }
            let mut rc = obj[j];
            for i in 0..n_rows {
                rc -= cb[i] * tab[i][j];
            }
            if rc > eps {
                entering = j as i64;
                break;
            }
        }
        if entering == -1 {
            return "optimal";
        }
        // ratio test, smallest-index tie-break on basis
        let mut leaving_row: i64 = -1;
        let mut min_ratio = INF;
        let mut min_basis_idx = (n_vars + 1) as i64;
        for i in 0..n_rows {
            let pivot = tab[i][entering as usize];
            if pivot > eps {
                let ratio = tab[i][rhs_col] / pivot;
                if ratio < min_ratio - eps
                    || ((ratio - min_ratio).abs() <= eps
                        && basis[i] < min_basis_idx)
                {
                    min_ratio = ratio;
                    leaving_row = i as i64;
                    min_basis_idx = basis[i];
                }
            }
        }
        if leaving_row == -1 {
            return "unbounded";
        }
        let lr = leaving_row as usize;
        let ent = entering as usize;
        let inv_pivot = 1.0 / tab[lr][ent];
        for k in 0..(n_vars + 1) {
            tab[lr][k] *= inv_pivot;
        }
        for i in 0..n_rows {
            if i == lr { continue; }
            let factor = tab[i][ent];
            if factor.abs() < eps { continue; }
            for k in 0..(n_vars + 1) {
                tab[i][k] -= factor * tab[lr][k];
            }
        }
        basis[lr] = entering;
        iters += 1;
        if iters >= max_iter {
            return "max_iter";
        }
    }
}

/// In-place simplex pivot.  Mutates ``tableau`` (list of lists of floats) and
/// ``basis`` (list of ints) exactly like the other ``_accel`` backends, and
/// returns only the status string.  Mirrors ``impl_python.run`` numerics.
#[pyfunction]
#[pyo3(signature = (tableau, basis, obj, n_vars, eps=1e-9, max_iter=10000, forbidden=None))]
fn run(tableau: &Bound<'_, PyAny>, basis: &Bound<'_, PyAny>, obj: Vec<f64>,
                n_vars: usize, eps: f64, max_iter: i64,
                forbidden: Option<Vec<i64>>) -> PyResult<String> {
    // Read tableau (list[list[float]]) into Rust, tracking dimensions.
    let n_rows = tableau.len()?;
    let mut tab: Vec<Vec<f64>> = Vec::with_capacity(n_rows);
    let cols = n_vars + 1;
    for i in 0..n_rows {
        let row = tableau.get_item(i)?;
        let mut r: Vec<f64> = Vec::with_capacity(cols);
        for k in 0..cols {
            r.push(row.get_item(k)?.extract::<f64>()?);
        }
        tab.push(r);
    }
    // Read basis.
    let nb = basis.len()?;
    let mut basis_i: Vec<i64> = Vec::with_capacity(nb);
    for i in 0..nb {
        basis_i.push(basis.get_item(i)?.extract::<i64>()?);
    }
    let mut forb: Vec<i64> = forbidden.unwrap_or_default();
    forb.sort_unstable();

    if n_rows == 0 {
        return Ok("optimal".to_string());
    }
    let status = simplex_run(&mut tab, &mut basis_i, &obj, n_vars, &forb, eps, max_iter);

    // Write results back in-place.
    for i in 0..n_rows {
        let row = tableau.get_item(i)?;
        for k in 0..cols {
            row.set_item(k, tab[i][k])?;
        }
    }
    for i in 0..nb {
        basis.set_item(i, basis_i[i])?;
    }
    Ok(status.to_string())
}

#[pymodule]
fn impl_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(simplex, m)?)?;
    m.add("_NATIVE_PRESENT", true)?;
    Ok(())
}

/// Convenience wrapper: copy-in/copy-out variant returning (status, tableau, basis).
#[pyfunction]
#[pyo3(signature = (tableau, basis, obj, n_vars, eps=1e-9, max_iter=10000, forbidden=None))]
fn simplex(tableau: Vec<Vec<f64>>, mut basis: Vec<i64>, obj: Vec<f64>, n_vars: usize,
           eps: f64, max_iter: i64, forbidden: Option<Vec<i64>>)
    -> PyResult<(String, Vec<Vec<f64>>, Vec<i64>)> {
    let n_rows = tableau.len();
    if n_rows == 0 {
        return Ok(("optimal".to_string(), tableau, basis));
    }
    let mut tab = tableau;
    let mut forb: Vec<i64> = forbidden.unwrap_or_default();
    forb.sort_unstable();
    let status = simplex_run(&mut tab, &mut basis, &obj, n_vars, &forb, eps, max_iter);
    Ok((status.to_string(), tab, basis))
}
