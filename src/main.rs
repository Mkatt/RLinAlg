use linalg::{Matrix, Vector};
use ndarray::{Array1, Array2};


fn main() {

    // Matrix Examples
    let a = Matrix {
        data: Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
    };
    let b = Matrix {
        data: Array2::from_shape_vec((2, 2), vec![4.0, 3.0, 2.0, 1.0]).unwrap(),
    };

    // Matrix Addition
    if let Ok(sum) = a.add(&b) {
        println!("Matrix Addition:\n{:?}", sum.data);
    }

    // Matrix Multiplication
    if let Ok(product) = a.multiply(&b) {
        println!("Matrix Multiplication:\n{:?}", product.data);
    }

    // Matrix Transpose
    let transpose = a.transpose();
    println!("Matrix Transpose:\n{:?}", transpose.data);

    // Matrix Determinant
    if let Some(determinant) = a.determinant() {
        println!("Matrix Determinant: {:?}", determinant);
    }

    // Matrix Inverse
    if let Ok(inverse) = a.inverse() {
        println!("Matrix Inverse:\n{:?}", inverse.data);
    }

    // Matrix LU Decomposition
    if let Ok((l, u)) = a.lu_decomposition() {
        println!("Matrix LU Decomposition - L:\n{:?}", l.data);
        println!("Matrix LU Decomposition - U:\n{:?}", u.data);
    }

    // Matrix Norms
    println!("Matrix L1 Norm: {:?}", a.l1_norm());
    println!("Matrix L2 Norm: {:?}", a.l2_norm());
    println!("Matrix Infinity Norm: {:?}", a.infinity_norm());

    // Matrix Trace
    println!("Matrix Trace: {:?}", a.trace());

    // Vector Examples
    let v1 = Vector {
        data: Array1::from_vec(vec![1.0, 2.0, 3.0]),
    };
    let v2 = Vector {
        data: Array1::from_vec(vec![4.0, 5.0, 6.0]),
    };

    // Vector Addition
    let v_sum = v1.add(&v2);
    println!("Vector Addition: {:?}", v_sum.data);

    // Vector Dot Product
    if let Ok(dot_product) = v1.dot(&v2) {
        println!("Vector Dot Product: {:?}", dot_product);
    }

    // Vector Magnitude
    println!("Vector Magnitude: {:?}", v1.magnitude());

    // Vector Normalization
    let normalized = v1.normalize();
    println!("Vector Normalization: {:?}", normalized.data);

    // Vector Norms
    println!("Vector L1 Norm: {:?}", v1.l1_norm());
    println!("Vector L2 Norm: {:?}", v1.l2_norm());

    // Eigenvector and Eigenvalue (if applicable)
    // Adjust the matrix to be suitable for eigenvector calculation
    let eigen_matrix = Matrix {
        data: Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 2.0]).unwrap(),
    };

    if let Ok(eigenvector) = eigen_matrix.eigenvector(1000, 1e-10) {
        println!("Eigenvector: {:?}", eigenvector.data);
        if let Ok(eigenvalue) = eigen_matrix.eigenvalue(&eigenvector) {
            println!("Corresponding Eigenvalue: {:?}", eigenvalue);
        }
    }

    // Kronecker Product Example
    let matrix1 = Matrix {
        data: Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
    };
    let matrix2 = Matrix {
        data: Array2::from_shape_vec((2, 2), vec![0.0, 5.0, 6.0, 7.0]).unwrap(),
    };

    let kronecker = matrix1.kronecker_product(&matrix2);
    println!("Kronecker Product:\n{:?}", kronecker.data);
    
}
