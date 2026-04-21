# Neural Dynamics CV Language

## GitHub Subtitle

Reproducible computational neuroscience project linking single-neuron spiking, adaptation, synchrony, and EEG-like neural-mass dynamics.

## 2-Line CV Summary

Built a reproducible computational neuroscience pipeline spanning integrate-and-fire neurons, adaptive spiking, coupled-neuron synchrony, and Jansen-Rit neural-mass dynamics. Quantified excitability, firing-rate suppression, synchrony changes, and spectral peaks with automated figures, tests, and CSV outputs.

## 2-Bullet CV Version

- Built a reproducible multi-scale neural dynamics pipeline and quantified excitability across reduced neuron models, measuring a rheobase shift from 2.0 nA to 1.5 nA and a 24.3% steady-rate reduction at 4.0 nA with dynamic thresholding.
- Implemented adaptation, synchrony, and neural-mass analyses with automated figure and CSV export, measuring a 75.35% adaptation-driven firing-rate reduction, a 2.0x synchrony increase, and a 3.0 Hz base PSD peak.

## 4-Bullet CV Version

- Refactored four neural modeling modules into a reusable Python project with command-line experiments, tests, and structured outputs, producing a reproducible end-to-end analysis workflow.
- Quantified single-neuron excitability in integrate-and-fire models, measuring rheobase values of 2.0 nA and 1.5 nA and a 17.20 Hz (24.3%) steady-rate reduction at 4.0 nA after adding threshold dynamics.
- Measured spike-frequency adaptation under sustained input, showing a steady-state firing-rate drop from 44.44 Hz to 10.95 Hz at 4.0 nA, a reduction of 33.49 Hz (75.35%).
- Simulated synaptic coupling and neural-mass dynamics, finding a 2.0x increase in coincidence fraction (0.286 -> 0.571) and a neural-mass spectral peak at 3.0 Hz with peak power rising from 1.449 to 16.838 across the coupling sweep.

## 60-Second Interview Explanation

This project started from several separate neural modeling assignments, and I turned them into one reproducible computational neuroscience study. The core idea was to follow neural dynamics across scales: first how a single neuron converts current into spikes, then how threshold recovery and slow adaptation suppress sustained firing, then how coupling changes synchrony between neurons, and finally how population interactions generate EEG-like rhythms in a neural mass model. I packaged the models into reusable code, added command-line experiments, saved figures and CSV metrics automatically, and wrote tests. The main scientific takeaway is that even simple reduced models can show interpretable transitions in excitability, synchrony, and oscillatory structure when you analyze them systematically.

## 120-Second Interview Explanation

I wanted the repository to show more than that I had completed separate exercises, so I reframed it as one mechanism-oriented project about neural dynamics across scales. The first part studies single-neuron excitability using integrate-and-fire models, where I measured how current maps to firing rate and how dynamic thresholding changes effective excitability. The second part adds slow conductance adaptation and quantifies how it suppresses sustained firing; at 4.0 nA, the steady-state firing rate drops by 75.35%, which gives a concrete measure of spike-frequency adaptation rather than just a qualitative figure. The third part moves to a two-neuron system with asymmetric synaptic coupling and measures synchrony with coincidence fraction and lag structure; in the parameter sweep, synchrony doubles from 0.286 to 0.571. The final part uses a Jansen-Rit neural mass model to generate EEG-like activity and estimate a dominant spectral peak, which in the base regime is 3.0 Hz. What I think is most valuable is not claiming novelty, but showing that I can build, organize, test, and interpret reduced computational models in a way that is reproducible and scientifically honest.

## Motivation Letter Sentences

- I developed a reproducible computational neuroscience project that links reduced models of single-neuron spiking, spike-frequency adaptation, coupled-neuron synchrony, and neural-mass EEG-like activity within one analysis pipeline.
- The project was important to me because I wanted to move beyond isolated coursework exercises and build a coherent modeling study that quantified how neural dynamics change across scales.
- Working on this repository strengthened both my scientific reasoning and my research software practice, including model refactoring, automated experiments, quantitative summaries, and honest interpretation of reduced-model results.

## Likely PI Questions And Sample Answers

### Why did you choose reduced models rather than more biophysically detailed ones?

I chose reduced models because the scientific goal here was mechanistic interpretability across scales rather than channel-level realism. Integrate-and-fire and neural-mass models let me isolate how threshold dynamics, adaptation, synaptic coupling, and population feedback affect measurable outputs such as firing rate, synchrony, and PSD peaks. I would treat this project as a foundation for later work with more detailed models, not as a substitute for them.

### What is the most scientifically meaningful result in the repo?

For me, the strongest result is that the same repository shows multiple distinct mechanisms for changing neural activity, and each one is quantified. Dynamic thresholding reduces steady-state firing by 24.3% at 4.0 nA, slow adaptation reduces it by 75.35%, and stronger excitatory coupling doubles the synchrony metric in the two-neuron model. That makes the project more than a coding exercise, because each module is tied to a concrete dynamical effect.

### What would you improve next if you extended this project?

The highest-value next step would be validation against literature expectations or benchmark regimes. For example, I would add a short comparison section explaining whether the single-neuron and Jansen-Rit behaviors match known qualitative regimes and where they differ. That would strengthen the transition from strong student project to stronger research-facing portfolio piece.
