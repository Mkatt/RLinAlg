use rayon::prelude::*;
use ndarray::{Array2, Array1, Zip, arr2, ArrayView1};
use nalgebra::{Matrix as NaMatrix, Vector as NaVector, Dim};


pub struct Matrix {
    data: Array2<f64>,
}
impl Matrix {
    pub fn add(&self, other: &Matrix) -> Matrix {
        if self.data.dim() != other.data.dim() {
            panic!("Matrices must be of the same dimensions");
        }

        let sum_data = &self.data + &other.data;
        Matrix { data: sum_data }
    }

    pub fn multiply(&self, other: &Matrix) -> Matrix {
        if self.data.ncols() != other.data.nrows() {
            panic!("Inner matrix dimensions must match for multiplication");
        }

        let n = self.data.nrows();
        let m = other.data.ncols();
        let p = self.data.ncols();


        let mut partial_results: Vec<Array2<f64>> = (0..n).into_par_iter()
            .map(|i| {
                let mut row_result = Array2::<f64>::zeros((1, m));
                for j in 0..m {
                    row_result[[0, j]] = (0..p).map(|k| self.data[[i, k]] * other.data[[k, j]]).sum();
                }
                row_result
            })
            .collect();

        let mut result = Array2::<f64>::zeros((n, m));
        for (i, partial) in partial_results.into_iter().enumerate() {
            result.row_mut(i).assign(&partial.row(0));
        }

        Matrix { data: result }
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

    pub(self) fn create_minor(&self, matrix: &Array2<f64>, row_to_exclude: usize, col_to_exclude: usize) -> Array2<f64> {
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
}

#[derive(Clone)]
pub struct Vector {
    data: Array1<f64>,
}
impl Vector {
    pub fn add(&self, other: &Vector) -> Vector {
        let self_slice = self.data.view();
        let other_slice = other.data.view();

        let sum_data = self_slice.as_slice().unwrap().par_iter()
                                .zip(other_slice.as_slice().unwrap().par_iter())
                                .map(|(&a, &b)| a + b)
                                .collect::<Vec<f64>>(); 

        Vector { data: Array1::from(sum_data) }
    }
    pub fn dot(&self, other: &Vector) -> f64 {
        if self.data.len() != other.data.len() {
            panic!("Vectors must be of the same length");
        }

        let self_slice = self.data.as_slice().unwrap();
        let other_slice = other.data.as_slice().unwrap();

        self_slice.par_iter()
                  .zip(other_slice.par_iter())
                  .map(|(&a, &b)| a * b)
                  .sum()
    }
    pub fn magnitude(&self) -> f64 {
        self.data.iter()
                 .map(|&x| x * x)
                 .sum::<f64>()
                 .sqrt()
    }

    pub fn normalize(&self) -> Vector {
        let mag = self.magnitude();
        if mag == 0.0 {
            return self.clone();
        }

        let normalized_data: Vec<f64> = self
            .data
            .iter()
            .map(|&x| x / mag)
            .collect();

        Vector {
            data: Array1::from(normalized_data),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::SQRT_2;

    #[test]
    fn test_matrix_addition() {
        let a = Matrix { data: Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap() };
        let b = Matrix { data: Array2::from_shape_vec((2, 2), vec![4.0, 3.0, 2.0, 1.0]).unwrap() };
        let result = a.add(&b);

        let expected = Array2::from_shape_vec((2, 2), vec![5.0, 5.0, 5.0, 5.0]).unwrap();
        assert_eq!(result.data, expected);
    }

    #[test]
    #[should_panic(expected = "Matrices must be of the same dimensions")]
    fn test_matrix_addition_panic() {
        let a = Matrix { data: Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap() };
        let b = Matrix { data: Array2::from_shape_vec((3, 2), vec![4.0, 3.0, 2.0, 1.0, 0.0, -1.0]).unwrap() };
        a.add(&b);
    }

    #[test]
    fn test_matrix_multiplication() {
        let a = Matrix { data: Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap() };
        let b = Matrix { data: Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 1.0, 2.0]).unwrap() };
        let result = a.multiply(&b);

        let expected = Array2::from_shape_vec((2, 2), vec![4.0, 4.0, 10.0, 8.0]).unwrap();
        assert_eq!(result.data, expected);
    }

    #[test]
    #[should_panic(expected = "Inner matrix dimensions must match for multiplication")]
    fn test_matrix_multiplication_panic() {
        let a = Matrix { data: Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap() };
        let b = Matrix { data: Array2::from_shape_vec((2, 2), vec![7.0, 8.0, 9.0, 10.0]).unwrap() };
        a.multiply(&b);
    }

    #[test]
    fn test_vector_addition() {
        let a = Vector { data: Array1::from_vec(vec![1.0, 2.0, 3.0]) };
        let b = Vector { data: Array1::from_vec(vec![4.0, 5.0, 6.0]) };
        let result = a.add(&b);

        let expected = Array1::from_vec(vec![5.0, 7.0, 9.0]);
        assert_eq!(result.data, expected);
    }

    #[test]
    fn test_vector_dot_product() {
        let a = Vector { data: Array1::from_vec(vec![1.0, 2.0, 3.0]) };
        let b = Vector { data: Array1::from_vec(vec![4.0, 5.0, 6.0]) };
        let result = a.dot(&b);

        let expected = 32.0; // 1*4 + 2*5 + 3*6
        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic(expected = "Vectors must be of the same length")]
    fn test_vector_dot_product_panic() {
        let a = Vector { data: Array1::from_vec(vec![1.0, 2.0, 3.0]) };
        let b = Vector { data: Array1::from_vec(vec![4.0, 5.0]) };
        a.dot(&b);
    }
    #[test]
    fn test_vector_magnitude_zero() {
        let v = Vector { data: Array1::from_vec(vec![0.0, 0.0, 0.0]) };
        let mag = v.magnitude();
        assert_eq!(mag, 0.0);
    }

    #[test]
    fn test_vector_magnitude_basic() {
        let v = Vector { data: Array1::from_vec(vec![3.0, 4.0]) };
        let mag = v.magnitude();
        assert_eq!(mag, 5.0); // 3-4-5 right triangle
    }

    #[test]
    fn test_vector_magnitude_negative_values() {
        let v = Vector { data: Array1::from_vec(vec![-3.0, -4.0]) };
        let mag = v.magnitude();
        assert_eq!(mag, 5.0); // Magnitude should be positive
    }

    #[test]
    fn test_vector_magnitude_fractional() {
        let v = Vector { data: Array1::from_vec(vec![1.0 / SQRT_2, 1.0 / SQRT_2]) };
        let mag = v.magnitude();
        assert!((mag - 1.0).abs() < 1e-10); // Should be close to 1 with a small error margin
    }

    #[test]
    fn test_vector_normalization() {
        let v = Vector {
            data: Array1::from_vec(vec![3.0, 4.0]),
        };
        let normalized = v.normalize();

        let expected = Array1::from_vec(vec![0.6, 0.8]);
        assert_eq!(normalized.data, expected);
    }

    #[test]
    fn test_matrix_transpose() {
        let a = Matrix {
            data: Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        };
        let transposed = a.transpose();

        let expected = Array2::from_shape_vec((3, 2), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();
        assert_eq!(transposed.data, expected);
    }

    #[test]
    fn test_matrix_determinant() {
        let a = Matrix {
            data: Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        };
        let determinant = a.determinant().unwrap();

        let expected = -2.0;
        assert_eq!(determinant, expected);
    }

    #[test]
    fn test_matrix_determinant_non_square() {
        let a = Matrix {
            data: Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        };
        let determinant = a.determinant();

        assert_eq!(determinant, None);
    }
}
