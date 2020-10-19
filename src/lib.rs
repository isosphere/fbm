//! Fractional Brownian Motion crate for Rust. An attempt to keep parity with the Python fpm module: https://github.com/crflynn/fbm
//! Code is largely directly taken from that Python module.

use itertools_num::ItertoolsNum;
use ndarray::prelude::*;
use ndarray_linalg::cholesky::*;
use rand::thread_rng;
use rand_distr::{Normal, Distribution};
use rustfft::FFTplanner;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;

pub enum Methods {
    DaviesHarte,
    Cholesky,
    Hosking
}

pub struct FBM {
    pub method: Methods,
    /// The number of increments to generate, frequently identified as 'n'
    pub increments: usize, 
    pub hurst: f64,
    pub length: f64, // also called "magnitude"
    // Values to speed up Monte Carlo simulation
    ds_eigenvals: Option<Vec<Complex<f64>>>,
    hosking_cov: Option<Vec<f64>>,
    cholesky_cov: Option<Array2<f32>>
}

impl FBM {
    pub fn new (method: Methods, increments: usize, hurst: f64, length: f64) -> Self {
        Self {
            method, increments, hurst, length,
            ds_eigenvals: None, hosking_cov: None, cholesky_cov: None
        }
    }

    fn autocovariance(&self, k: usize) -> Complex<f64> {
        let k = k as f64;
        Complex::zero() + 0.5 * ( (k - 1.0).abs().powf(2.0 * self.hurst) - 2.0 * k.abs().powf(2.0 * self.hurst) + (k + 1.0).abs().powf(2.0 * self.hurst) )
    }

    /// Generate a fgn realization using Davies-Harte method.
    /// Uses Davies and Harte method (exact method) from:
    /// Davies, Robert B., and D. S. Harte. "Tests for Hurst effect."
    /// Biometrika 74, no. 1 (1987): 95-101.
    /// Can fail if n is small and hurst close to 1. 
    /// See:
    /// Wood, Andrew TA, and Grace Chan. "Simulation of stationary Gaussian
    /// processes in [0, 1] d." Journal of computational and graphical
    /// statistics 3, no. 4 (1994): 409-432.
    fn daviesharte(&mut self, gn: Vec<f64>) -> Vec<f64> {
        // Monte carlo consideration
        if self.ds_eigenvals.is_none() {
            // Generate the first row of the circulant matrix
            let row_component: Vec<Complex<f64>> = (1 .. self.increments).map(|i| self.autocovariance(i) ).collect();
            let mut reverse_component = row_component.clone();
            reverse_component.reverse();

            let mut row = [
                vec![Complex::zero(), self.autocovariance(0)], 
                row_component, 
                vec![Complex::zero()], 
                reverse_component
            ].concat();

            // Get the eigenvalues of the circulant matrix. 
            // Discard the imaginary part (should all be zero in theory so imaginary part will be very small)
            let mut planner = FFTplanner::new(false);
            let fft = planner.plan_fft(row.len());
            
            self.ds_eigenvals = {
                let mut eigenvals = vec![Complex::zero(); row.len()];

                self.ds_eigenvals = Some(Vec::new());
                fft.process(&mut row, eigenvals.as_mut_slice());
                Some(eigenvals)
            };
        }

        // If any of the eigenvalues are negative, then the circulant matrix
        // is not positive definite, meaning we cannot use this method. This
        // occurs for situations where n is low and H is close to 1.
        // See the following for a more detailed explanation:
        // Wood, Andrew TA, and Grace Chan. "Simulation of stationary Gaussian
        //     processes in [0, 1] d." Journal of computational and graphical
        //     statistics 3, no. 4 (1994): 409-432.
        if self.ds_eigenvals.as_ref().unwrap().iter().any(|ev| ev.re.is_sign_negative()) {
            panic!(
                "Found a negative eigenvalue. Combination of increments n={} and Hurst parameter={} invalid for Davies-Harte method.
                Occurs when n is small and Hurst is close to 1. Use the Hosking method.", self.increments, self.hurst
            )
        }

        // Generate second sequence of i.i.d. standard normals
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let gn2: Vec<Complex<f64>> = (0 .. self.increments).map(|_i| Complex{re: normal.sample(&mut rng), im:0.0}).collect();

        // Resulting sequence from matrix multiplication of positive definite
        // sqrt(C) matrix with fgn sample can be simulated in this way.
        //w = np.zeros(2 * self.n, dtype=complex)
        let mut w: Vec<Complex<f64>> = Vec::new();
        for i in 0 .. 2 * self.increments {
            let eignvalue = match self.ds_eigenvals.as_ref() {
                Some(v) => {v[i]},
                None => panic!("Expected eigenvalue at {}, got None", i)
            };

            match i {
                i if i == 0 => {
                    w.push((eignvalue/ (2.0 * self.increments as f64)).sqrt() * gn[i]);
                },
                i if i < self.increments => {
                    let left_side = (eignvalue / (4.0 * self.increments as f64)).sqrt();
                    let right_side: Complex<f64> = Complex{re: gn[i], im: 0.0} + gn2[i];
                    w.push(left_side * right_side);
                },
                i if i == self.increments => {
                    w.push((eignvalue / (2.0 * self.increments as f64)).sqrt() * gn2[0]);
                },
                _ => {
                    let left_side = (eignvalue / (4.0 * self.increments as f64)).sqrt();
                    let right_side: Complex<f64> = gn[2 * self.increments - i] - gn2[2 * self.increments - i];
                    w.push(left_side * right_side);
                }
            }
        }

        // Resulting z is fft of sequence w. Discard small imaginary part (z should be real in theory).
        let mut planner = FFTplanner::new(false);
        let fft = planner.plan_fft(w.len());

        let z = {
            let mut fft_values = vec![Complex::zero(); w.len()];
            fft.process(&mut w, fft_values.as_mut_slice());
            fft_values
        };

        (0 .. self.increments).map(|i| z[i].re).collect()
    }

    /// Generate a fgn realization using the Cholesky method.

    ///     Uses Cholesky decomposition method (exact method) from:
    ///     Asmussen, S. (1998). Stochastic simulation with a view towards
    ///     stochastic processes. University of Aarhus. Centre for Mathematical
    ///     Physics and Stochastics (MaPhySto)[MPS].
    fn cholesky(&mut self, gn: Vec<f64>) -> Vec<f32> {
        if self.cholesky_cov.is_none() {
            let mut covariance: Array2<f32> = ArrayBase::zeros((self.increments, self.increments));
            
            for ((i, j), value) in covariance.indexed_iter_mut() {
                *value = self.autocovariance(i - j).re as f32; 
            }

            println!("autocovariance set");

            self.cholesky_cov = Some(covariance.cholesky(UPLO::Lower).unwrap());
        }

        let truncated_gn = {
            let gn_downsample = gn.iter().map(|v| *v as f32).collect::<Vec<f32>>();
            Array1::from(gn_downsample)
        };

        self.cholesky_cov.as_ref().unwrap().dot(&truncated_gn.t()).to_vec()
    }

    /// Generate a fGn realization using Hosking's method.
    /// Method of generation is Hosking's method (exact method) from his paper:
    /// Hosking, J. R. (1984). Modeling persistence in hydrological time series
    /// using fractional differencing. Water resources research, 20(12),
    /// 1898-1908.
    fn hosking(&mut self, gn: Vec<f64>) -> Vec<f64> {
        let mut fgn = vec![0.0; self.increments];
        let mut phi = vec![0.0; self.increments];
        let mut psi = vec![0.0; self.increments];

        // Monte carlo consideration
        if self.hosking_cov.is_none() {
            self.hosking_cov = Some((0 .. self.increments).map(|i| self.autocovariance(i).re).collect());
        }

        // First increment from stationary distribution
        fgn[0] = gn[0];
        phi[0] = 0.0;

        let mut v = 1.0;

        // Generate fgn realization with n increments of size 1
        let cov = self.hosking_cov.as_ref().unwrap();

        for i in 1 .. self.increments {
            phi[i - 1] = cov[i];
            for j in 0 .. i - 1 {
                psi[j] = phi[j];
                phi[i - 1] -= psi[j] * cov[i - j - 1];
            }
            phi[i - 1] /= v;
            for j in 0 .. i - 1 {
                phi[j] = psi[j] - phi[i - 1] * psi[i - j - 2];
            }
            v *= 1.0 - phi[i - 1] * phi[i - 1];
            for j in 0 .. i {
                fgn[i] += phi[j] * fgn[i - j - 1];
            }
            fgn[i] += v.sqrt() * gn[i];
        }

        fgn
    }

    /// Sample the fractional Brownian motion
    pub fn fbm(&mut self) -> Vec<f64> {
        let mut sampled: Vec<f64> = self.fgn().iter().cumsum().collect();
        sampled.push(0.0); // why?
        sampled
    }

    /// Sample the fractional Gaussian noise
    pub fn fgn(&mut self) -> Vec<f64> {
        let scale = (self.length / self.increments as f64).powf(self.hurst);
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        
        let gn: Vec<f64> = (0 .. self.increments).map(|_i| normal.sample(&mut rng)).collect();

        if (self.hurst - 0.5).abs() < f64::EPSILON {
            gn.iter().map(|e| scale*e).collect()
        } else {
            match &self.method {
                Methods::DaviesHarte => {
                    self.daviesharte(gn).iter().map(|e| e*scale).collect()
                },
                Methods::Hosking => {
                    self.hosking(gn).iter().map(|e| e*scale).collect()
                },
                Methods::Cholesky => {
                    panic!("Not implemented: fails with memory allocation errors");
                    //self.cholesky(gn).iter().map(|e| (*e as f64)*scale).collect()
                },
            }
        }
    }
}