import numpy as np
from mapspec.spectrum import TextSpec
from matplotlib import pyplot as plt
import sys, os

class main(object):
	def __init__(self, filename, path):
		self.filename = filename
		self.path = path
		        		
	def sub_main(self):
		spectra, snr = np.genfromtxt(self.filename, usecols = (0, 1), skip_header = 1, unpack = True, dtype = str)
		spectra = np.array([self.path + f for f in spectra])
		snr = np.array([float(num) for num in snr])
		
		mask = snr > 10 # Below this, is considered bad data
		
		spectra, snr = spectra[mask], snr[mask]
    
		print('Number of input spectra: %s' % (len(spectra)))
		
		self.spectra = spectra
		self.template_index = np.argmax(snr)
		self.align_min = -200
		self.align_max = 200
		self.step = 0.1
		
		def readSpec(specfile):
    
    			try:
        			spec = TextSpec(specfile)
    
    			except:
        			message = 'Error reading from file %s' % (specfile)
        			print(message)
    
    			return spec

		def crosscorr(w, f, tw, tf, align_min, align_max, step):

    			dws = np.arange(align_min, align_max + step, step)
    			cc = np.zeros(np.size(dws))

    			for idx, dw in enumerate(dws):

        			m = (w - dw >= w[0]) & (w - dw <= w[-1])

        			mtf = np.interp(w[m] - dw, tw, tf)
        			mf = f[m]

        			mtf -= np.mean(mtf)
        			mf -= np.mean(mf)

        			cc[idx] = np.sum(mtf * mf) / (np.size(mf) * np.std(mf) * np.std(mtf))

    			maxind = np.argmax(cc)

    			return dws[maxind]

		def get_shifts(self):

			shifts = []
			template = self.spectra[self.template_index]
			
			for spectrum in self.spectra:
        
				s, t = readSpec(spectrum), readSpec(template)

				shift = crosscorr(s.wv, s.f, t.wv, t.f, self.align_min, self.align_max, self.step)
				shifts.append(shift)

			return shifts

		def apply_shifts(spectra, shifts):

    			fs = []
    			ws = []
    			es = []

    			for idx, spectrum in enumerate(spectra):
        			spec = readSpec(spectrum)

        			shifted_f = np.interp(spec.wv + shifts[idx], spec.wv, spec.f)
        			shifted_e = np.interp(spec.wv + shifts[idx], spec.wv, spec.ef)
        
        			ws.append(spec.wv)
        			fs.append(shifted_f)
        			es.append(shifted_e)

    			return ws, fs, es

		def get_mean_spec(ws, fs, es):

    			dxs = [ws[idx][1] - ws[idx][0] for idx in range(len(ws))]
    			dx = np.mean(dxs)

    			wgrid = np.arange(ws[0][0], ws[0][-1] + dx, dx)

    			fi = [np.interp(wgrid, ws[idx], fs[idx]) for idx in range(len(ws))]
    			ei = [np.interp(wgrid, ws[idx], es[idx]) for idx in range(len(ws))]
    
    			fmean = np.mean(fi, axis = 0)
    			emean = np.mean(ei, axis = 0)

    			return wgrid, fmean, emean

		def extract_region(specfile, wavelength_region):
    
    			spec = readSpec(specfile)
    			m = (spec.wv >= wavelength_region[0]) & (spec.wv <= wavelength_region[1])
    			w, f, e = spec.wv[m], spec.f[m], spec.ef[m]
    
    			outfile = os.path.splitext(specfile)[0] + '.dat'
    			np.savetxt(outfile, np.transpose([w, f, e]))

		def plot_shifted_spectra(ws, fs):

    			for idx in range(len(ws)):

        			plt.plot(ws[idx], fs[idx])

		def saveSpectra(specfiles, ws, fs, es):

    			for idx in range(len(ws)):
        			outfile = os.path.splitext(specfiles[idx])[0] + '_aligned.dat'
        			np.savetxt(outfile, np.transpose([ws[idx], fs[idx], es[idx]]))
    		
		shifts = get_shifts(self)
		ws, fs, es = apply_shifts(spectra, shifts)
		wm, fm, em = get_mean_spec(ws, fs, es)
    		
		save = True
		if save:
			saveSpectra(spectra, ws, fs, es)
			np.savetxt('ref.txt', np.transpose([wm, fm, em]))
    
		plot_shifted_spectra(ws, fs)

infile = sys.argv[1] # Two column file containing spectral names and signal-to-noise ratio
path = sys.argv[2] # Path or directory of the input files in infile

run = main(infile, path)
run.sub_main()

plt.show()
