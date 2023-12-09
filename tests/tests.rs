use linalg::Matrix;
use linalg::Vector;
use ndarray::{Array1, Array2};
#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::SQRT_2;

    fn assert_matrix_eq(a: &Array2<f64>, b: &Array2<f64>, tol: f64) {
        assert!(a.iter().zip(b.iter()).all(|(&x, &y)| (x - y).abs() < tol));
    }
    #[test]
    fn test_matrix_addition() {
        let a = Matrix {
            data: Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        };
        let b = Matrix {
            data: Array2::from_shape_vec((2, 2), vec![4.0, 3.0, 2.0, 1.0]).unwrap(),
        };
        let result = a.add(&b).unwrap();

        let expected = Array2::from_shape_vec((2, 2), vec![5.0, 5.0, 5.0, 5.0]).unwrap();
        assert_eq!(result.data, expected);
    }


    #[test]
    fn test_matrix_multiplication() {
        let a = Matrix {
            data: Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        };
        let b = Matrix {
            data: Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 1.0, 2.0]).unwrap(),
        };
        let result = a.multiply(&b).unwrap();

        let expected = Array2::from_shape_vec((2, 2), vec![4.0, 4.0, 10.0, 8.0]).unwrap();
        assert_eq!(result.data, expected);
    }


    #[test]
    fn test_vector_addition() {
        let a = Vector {
            data: Array1::from_vec(vec![1.0, 2.0, 3.0]),
        };
        let b = Vector {
            data: Array1::from_vec(vec![4.0, 5.0, 6.0]),
        };
        let result = a.add(&b);

        let expected = Array1::from_vec(vec![5.0, 7.0, 9.0]);
        assert_eq!(result.data, expected);
    }

    #[test]
    fn test_vector_dot_product() {
        let a = Vector {
            data: Array1::from_vec(vec![1.0, 2.0, 3.0]),
        };
        let b = Vector {
            data: Array1::from_vec(vec![4.0, 5.0, 6.0]),
        };
        let result = a.dot(&b).unwrap();

        let expected = 32.0; // 1*4 + 2*5 + 3*6
        assert_eq!(result, expected);
    }


    #[test]
    fn test_vector_magnitude_zero() {
        let v = Vector {
            data: Array1::from_vec(vec![0.0, 0.0, 0.0]),
        };
        let mag = v.magnitude();
        assert_eq!(mag, 0.0);
    }

    #[test]
    fn test_vector_magnitude_basic() {
        let v = Vector {
            data: Array1::from_vec(vec![3.0, 4.0]),
        };
        let mag = v.magnitude();
        assert_eq!(mag, 5.0); // 3-4-5 right triangle
    }

    #[test]
    fn test_vector_magnitude_negative_values() {
        let v = Vector {
            data: Array1::from_vec(vec![-3.0, -4.0]),
        };
        let mag = v.magnitude();
        assert_eq!(mag, 5.0); // Magnitude should be positive
    }

    #[test]
    fn test_vector_magnitude_fractional() {
        let v = Vector {
            data: Array1::from_vec(vec![1.0 / SQRT_2, 1.0 / SQRT_2]),
        };
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
    #[test]
    fn test_matrix_inverse() {
        let a = Matrix {
            data: Array2::from_shape_vec((2, 2), vec![4.0, 7.0, 2.0, 6.0]).unwrap(),
        };
        let inverse = a.inverse().unwrap();

        let expected = Array2::from_shape_vec((2, 2), vec![0.6, -0.7, -0.2, 0.4]).unwrap();
        assert_matrix_eq(&inverse.data, &expected, 1e-10);
    }

    #[test]
    #[should_panic(expected = "Matrix is not invertible")]
    fn test_matrix_inverse_non_invertible() {
        let a = Matrix {
            data: Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 2.0, 4.0]).unwrap(),
        };
        a.inverse().unwrap();
    }

    #[test]
    fn test_matrix_l1_norm() {
        let a = Matrix {
            data: Array2::from_shape_vec((2, 2), vec![1.0, -2.0, 3.0, -4.0]).unwrap(),
        };
        let norm = a.l1_norm();

        let expected = 10.0; // |1| + |-2| + |3| + |-4|
        assert_eq!(norm, expected);
    }

    #[test]
    fn test_matrix_l2_norm() {
        let a = Matrix {
            data: Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        };
        let norm = a.l2_norm();

        let expected = 5.477225575051661; // sqrt(1^2 + 2^2 + 3^2 + 4^2)
        assert_eq!(norm, expected);
    }

    #[test]
    fn test_matrix_infinity_norm() {
        let a = Matrix {
            data: Array2::from_shape_vec((2, 2), vec![1.0, -2.0, -3.0, 4.0]).unwrap(),
        };
        let norm = a.infinity_norm();

        let expected = 7.0; // max(|1 - 2|, |-3 + 4|)
        assert_eq!(norm, expected);
    }
    #[test]
    fn test_matrix_trace() {
        let a = Matrix {
            data: Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        };
        let trace = a.trace();

        let expected = 5.0; // 1 + 4
        assert_eq!(trace, expected);
    }
    #[test]
    fn test_vector_l1_norm() {
        let v = Vector {
            data: Array1::from_vec(vec![1.0, -2.0, 3.0, -4.0]),
        };
        let norm = v.l1_norm();

        let expected = 10.0; // |1| + |-2| + |3| + |-4|
        assert_eq!(norm, expected);
    }
    #[test]
    fn test_vector_l2_norm() {
        let v = Vector {
            data: Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]),
        };
        let norm = v.l2_norm();

        let expected = 5.477225575051661; // sqrt(1^2 + 2^2 + 3^2 + 4^2)
        assert!((norm - expected).abs() < 1e-10);
    }

    #[test]
    fn test_power_iteration_eigenvector() {
        let a = Matrix {
            data: Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 2.0]).unwrap(),
        };
        let eigenvector = a.eigenvector(1000, 1e-10).unwrap();

        let expected = Vector {
            data: Array1::from_vec(vec![1.0 / SQRT_2, 1.0 / SQRT_2]),
        };
        assert_vector_eq(&eigenvector.data, &expected.data, 1e-10);
    }

    #[test]
    fn test_rayleigh_quotient_eigenvalue() {
        let a = Matrix {
            data: Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 2.0]).unwrap(),
        };
        let eigenvector = Vector {
            data: Array1::from_vec(vec![1.0 / SQRT_2, 1.0 / SQRT_2]),
        };
        let eigenvalue = a.eigenvalue(&eigenvector).unwrap();

        let expected = 3.0; // Expected dominant eigenvalue for this matrix
        assert!((eigenvalue - expected).abs() < 1e-10);
    }

    // Helper function for comparing vectors
    fn assert_vector_eq(a: &Array1<f64>, b: &Array1<f64>, tol: f64) {
        assert!(a.iter().zip(b.iter()).all(|(&x, &y)| (x - y).abs() < tol));
    }
}
