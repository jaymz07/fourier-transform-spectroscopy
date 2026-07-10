# fourier-transform-spectroscopy
A specialized Python toolkit for processing raw interferograms from Michelson-type Fourier Transform Spectrometers (FTS) into high-resolution optical power spectra. This repository implements advanced numerical strategies to handle non-linear translation stage velocities and extreme dynamic range requirements.

## Key Features

* **Velocity-Invariant Processing:** Unlike standard FFT approaches that suffer from spectral broadening due to motor instabilities, this toolkit utilizes a re-sampling algorithm (Cubic Spline Interpolation) to map signals onto a uniform Optical Path Difference ($\delta$) grid.
* **Dynamic Range Stitching:** This allows users to merge data from two simultaneous acquisition channels (High-Gain and Attenuated), effectively bypassing the ADC's dynamic range limitations during centerburst saturation.
* **Reference Laser Integration:** Support for utilizing high-resolution reference laser interferograms to track instantaneous displacement, providing a precise temporal map for signal re-sampling.

## Repository Structure

| File | Description |
| :--- | :--- |
| `FTIR_Functions.py` | The core computational engine containing the interpolation and Fourier transform algorithms. |
| `Retrieve_Spectrum.ipynb` | A standard implementation workflow for single-channel, constant-velocity interferograms. |
| `Retrieve_Spectrum_Stitched.ipynb` | An advanced workflow demonstrating the dual-channel stitching process with logistic crossover. |
| `FTS_Theory.pdf` | Detailed mathematical derivation of the Michelson interference and the stitching algorithm. |
| `tex/` | LaTeX source files for the theoretical documentation and technical notes. |

## 🛠 Mathematical Foundation

The processing logic is based on the principle that the detected intensity $I(t)$ is a modulated signal:
$$I(t) = I_0 + \int_{0}^{\infty} |\mathcal{A}(\omega)|^2 \cos\left(\frac{2\omega v t}{c} + \phi_0\right) d\omega$$

To handle non-uniform scanning velocity $v(t)$, the software performs a transformation from the time domain ($t$) to the optical path difference domain ($\delta$), ensuring that the subsequent Fourier Transform is performed on a strictly uniform grid.

For high-amplitude signals, we implement **Dynamic Range Stitching**. When two signals (High-Gain $I_H$ and Low-Gain $I_L$) are available, they are merged using a sigmoid weight $w(t)$:
$$I_{\text{stitched}} = [1 - w(t)] I_H + w(t) I_{L,\text{scaled}}$$
This prevents the introduction of discontinuities that would otherwise manifest as high-frequency noise in the recovered spectrum.

## Usage Guide

### Prerequisites
* Python 3
* `numpy`
* `scipy`
* `matplotlib`
* `jupyter`

### Standard Workflow
1. Prepare your raw interferogram data (Time-series and, optionally, a Reference Laser displacement map).
2. Open `Retrieve_Spectrum.ipynb`.
3. Configure the interpolation parameters and execute the cells to generate the power spectrum.

### High-Dynamic Range (HDR) Workflow
Use a BNC Tee to split your signal into two channels. The gain of the second channel should be adjusted to give saturation of the centurburst, but not of the small oscillations occuring at larger delays. The script should be adjusted to read the two scope channels into the appropriate labels.

## 📜 Citation & Documentation
The full mathematical derivation for the interpolation and stitching logic can be found in `FTS_Theory.pdf` within this repository.
