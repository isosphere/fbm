//! Fractional Brownian Motion crate for Rust. An attempt to keep parity with the Python fpm module: https://github.com/crflynn/fbm
//! Code is largely directly taken from that Python module.

use itertools_num::ItertoolsNum;
use rand::thread_rng;
use rand_distr::{Normal, Distribution};
use rustfft::FFTplanner;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;

pub enum Methods {
    DaviesHarte,
//    Cholesky, // TODO not implemented
//    Hosking   // TODO not implemented
}

pub struct FBM {
    pub method: Methods,
    /// The number of increments to generate, frequently identified as 'n'
    pub increments: usize, 
    pub hurst: f64,
    pub length: u64,
    // Values to speed up Monte Carlo simulation
    autocovariance: Option<Vec<Complex<f64>>>,
    eigenvals: Option<Vec<Complex<f64>>>,
}

impl FBM {
    pub fn new (method: Methods, increments: usize, hurst: f64, length: u64) -> Self {
        Self {
            method, increments, hurst, length,
            autocovariance: None, eigenvals: None
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
        if self.eigenvals.is_none() {
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
            
            self.eigenvals = {
                let mut eigenvals = Vec::new();

                self.eigenvals = Some(Vec::new());
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
        if self.eigenvals.as_ref().unwrap().iter().filter(|ev| ev.re < 0.0).collect::<Vec<&Complex<f64>>>().len() > 0 {
            panic!(
                "Combination of increments n and Hurst value H invalid for Davies-Harte method.
                Occurs when n is small and Hurst is close to 1. "
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
            let eignvalue = match self.eigenvals.as_ref() {
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
            let mut fft_values = Vec::new();
            fft.process(&mut w, fft_values.as_mut_slice());
            fft_values
        };

        (0 .. self.increments).map(|i| z[i].re).collect()
    }

    /// Sample the fractional Brownian motion
    pub fn fbm(&mut self) -> Vec<f64> {
        let mut sampled: Vec<f64> = self.fgn().iter().cumsum().collect();
        sampled.push(0.0); // why?
        sampled
    }

    /// Sample the fractional Gaussian noise
    pub fn fgn(&mut self) -> Vec<f64> {
        let scale = (self.length as f64 / self.increments as f64).powf(self.hurst);
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        
        let gn: Vec<f64> = (0 .. self.increments).map(|_i| normal.sample(&mut rng)).collect();

        if (self.hurst - 0.5).abs() < f64::EPSILON {
            gn.iter().map(|e| scale*e).collect()
        } else {
            match &self.method {
                Methods::DaviesHarte => {
                    self.daviesharte(gn).iter().map(|e| e*scale).collect()
                }
                _ => {
                    panic!("Method unsupported.")
                }
            }
        }
    }
}