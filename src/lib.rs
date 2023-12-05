use ndarray::{Array1, Array2};
use rayon::prelude::*;

pub struct Matrix {
    pub data: Array2<f64>,
}
impl Matrix {
    pub fn add(&self, other: &Matrix) -> Result<Matrix, String> {
        if self.data.dim() != other.data.dim() {
            return Err("Matrices must be of the same dimensions".to_string());
        }

        let sum_data = &self.data + &other.data;
        Ok(Matrix { data: sum_data })
    }

    pub fn multiply(&self, other: &Matrix) -> Result<Matrix, String> {
        if self.data.ncols() != other.data.nrows() {
            return Err("Inner matrix dimensions must match for multiplication".to_string());
        }

        let n = self.data.nrows();
        let m = other.data.ncols();
        let p = self.data.ncols();

        let partial_results: Vec<Array2<f64>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut row_result = Array2::<f64>::zeros((1, m));
                for j in 0..m {
                    row_result[[0, j]] =
                        (0..p).map(|k| self.data[[i, k]] * other.data[[k, j]]).sum();
                }
                row_result
            })
            .collect();

        let mut result = Array2::<f64>::zeros((n, m));
        for (i, partial) in partial_results.into_iter().enumerate() {
            result.row_mut(i).assign(&partial.row(0));
        }

        Ok(Matrix { data: result })
    }

    pub fn transpose(&self) -> Matrix {
        Matrix {
            data: self.data.t().to_owned(),
        }
    }

    pub fn determinant(&self) -> Option<f64> {
        let (rows, cols) = self.data.dim();
        if rows != cols {
            return None; // Not a square matrix
        }

        Some(self.calculate_determinant(&self.data))
    }

    pub(self) fn calculate_determinant(&self, matrix: &Array2<f64>) -> f64 {
        let (rows, _) = matrix.dim();

        if rows == 1 {
            return matrix[[0, 0]];
        }

        let mut determinant = 0.0;
        let mut sign = 1.0;

        for col in 0..rows {
            let minor = self.create_minor(matrix, 0, col);
            determinant += sign * matrix[[0, col]] * self.calculate_determinant(&minor);
            sign *= -1.0;
        }

        determinant
    }

    pub(self) fn create_minor(
        &self,
        matrix: &Array2<f64>,
        row_to_exclude: usize,
        col_to_exclude: usize,
    ) -> Array2<f64> {
        let (rows, cols) = matrix.dim();
        let mut minor = Array2::<f64>::zeros((rows - 1, cols - 1));

        let mut minor_row = 0;
        let mut minor_col;

        for row in 0..rows {
            if row == row_to_exclude {
                continue;
            }
            minor_col = 0;
            for col in 0..cols {
                if col != col_to_exclude {
                    minor[[minor_row, minor_col]] = matrix[[row, col]];
                    minor_col += 1;
                }
            }
            minor_row += 1;
        }

        minor
    }

    pub fn identity(size: usize) -> Matrix {
        let mut data = Array2::<f64>::zeros((size, size));
        for i in 0..size {
            data[[i, i]] = 1.0;
        }
        Matrix { data }
    }

    pub fn zero(rows: usize, cols: usize) -> Matrix {
        let data = Array2::<f64>::zeros((rows, cols));
        Matrix { data }
    }

    pub fn inverse(&self) -> Result<Matrix, String> {
        let (rows, cols) = self.data.dim();

        if rows != cols {
            return Err("Only square matrices can be inverted".to_string());
        }

        let det = self.determinant().unwrap_or(0.0);

        if det == 0.0 {
            return Err("Matrix is not invertible".to_string());
        }

        let cofactors: Vec<_> = (0..rows)
            .into_par_iter()
            .flat_map(|i| {
                (0..cols).into_par_iter().map(move |j| {
                    let minor = self.create_minor(&self.data, i, j);
                    let cofactor =
                        self.calculate_determinant(&minor) * (-1.0f64).powi((i + j) as i32);
                    (i, j, cofactor)
                })
            })
            .collect();

        let mut adjugate = Array2::<f64>::zeros((rows, cols));
        for (i, j, cofactor) in cofactors {
            adjugate[[j, i]] = cofactor;
        }

        let inverse_data = adjugate.mapv(|x| x / det);

        Ok(Matrix { data: inverse_data })
    }

    pub fn lu_decomposition(&self) -> Result<(Matrix, Matrix), String> {
        let n = self.data.nrows();
        if n != self.data.ncols() {
            return Err("Matrix must be square".to_string());
        }

        let mut lower = Array2::<f64>::zeros((n, n));
        let mut upper = Array2::<f64>::zeros((n, n));

        for i in 0..n {
            for k in i..n {
                let sum = (0..i).fold(0.0, |sum, j| sum + lower[[i, j]] * upper[[j, k]]);
                upper[[i, k]] = self.data[[i, k]] - sum;
            }

            for k in i..n {
                if i == k {
                    lower[[i, i]] = 1.0;
                } else {
                    let sum = (0..i).fold(0.0, |sum, j| sum + lower[[k, j]] * upper[[j, i]]);
                    lower[[k, i]] = (self.data[[k, i]] - sum) / upper[[i, i]];
                }
            }
        }

        Ok((Matrix { data: lower }, Matrix { data: upper }))
    }

    pub fn l1_norm(&self) -> f64 {
        self.data.iter().map(|&x| x.abs()).sum()
    }

    pub fn l2_norm(&self) -> f64 {
        self.data.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    pub fn infinity_norm(&self) -> f64 {
        self.data
            .axis_iter(ndarray::Axis(0))
            .map(|row| row.iter().map(|&x| x.abs()).sum::<f64>())
            .max_by(|x, y| x.partial_cmp(y).unwrap())
            .unwrap_or(0.0)
    }

    pub fn trace(&self) -> f64 {
        self.data.diag().iter().sum()
    }
}

#[derive(Clone)]
pub struct Vector {
    pub data: Array1<f64>,
}
impl Vector {
    pub fn add(&self, other: &Vector) -> Vector {
        let self_slice = self.data.view();
        let other_slice = other.data.view();

        let sum_data = self_slice
            .as_slice()
            .unwrap()
            .par_iter()
            .zip(other_slice.as_slice().unwrap().par_iter())
            .map(|(&a, &b)| a + b)
            .collect::<Vec<f64>>();

        Vector {
            data: Array1::from(sum_data),
        }
    }
    pub fn dot(&self, other: &Vector) -> Result<f64, String> {
        if self.data.len() != other.data.len() {
            return Err("Vectors must be of the same length".to_string());
        }

        let self_slice = self.data.as_slice().unwrap();
        let other_slice = other.data.as_slice().unwrap();

        let dot_product = self_slice
            .par_iter()
            .zip(other_slice.par_iter())
            .map(|(&a, &b)| a * b)
            .sum();

        Ok(dot_product)
    }

    pub fn magnitude(&self) -> f64 {
        self.data.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    pub fn normalize(&self) -> Vector {
        let mag = self.magnitude();
        if mag == 0.0 {
            return self.clone();
        }

        let normalized_data: Vec<f64> = self.data.iter().map(|&x| x / mag).collect();

        Vector {
            data: Array1::from(normalized_data),
        }
    }

    pub fn l1_norm(&self) -> f64 {
        self.data.par_iter().map(|&x| x.abs()).sum()
    }

    pub fn l2_norm(&self) -> f64 {
        self.data.par_iter().map(|&x| x * x).sum::<f64>().sqrt()
    }
}
