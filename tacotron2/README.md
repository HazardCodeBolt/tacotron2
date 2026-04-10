# Tacotron2: Autoregressive Spectrogram Generation


<img src="https://github.com/priyammaz/PyTorch-Adventures/blob/main/src/visuals/tacotron2_diagram.png?raw=true" alt="drawing" width="400"/>


One of the first stages of many Text-to-Speech (TTS) systems is to take text and conditionally generate a Mel Spectrogram. We can then take this magnitude spectrogram and use Griffin-Lim, or fancier Vocoders like HifiGAN, to then synthesize audio. Tacotron2 is one such early example, using a Sequence to Sequence model along with Location Sensitive Attention to encourage learning the alignment of Text and its cooresponding Mel Spectrogram. 

### References:

I referenced the following repos to implement this!
- [NVIDIA/tacotron2](https://github.com/NVIDIA/tacotron2/tree/master)
- [kaituoxu/Tacotron2](https://github.com/kaituoxu/Tacotron2/tree/master)


### Dataset

We will be training on the standard [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/) that you can download for free! It is about 25 hours of high quality audio from a single speaker. We can quickly create a train/test split for this dataset using the ```prep_splits.py``` file! This will produce a ```train_metadata.csv``` and a ```test_metadata.csv```

```bash
python prep_splits.py 
    --path_to_ljspech <PATH_TO_DOWNLOADED_DATA_DIR> \
    --path_to_save <PATH_TO_SAVE_DIR>
```

### Train Model

We will be training a standard Tacotron2 as described in the original paper. You can submit a training job with the following:

```bash
accelerate launch train_taco.py \
    --experiment_name tacotron2 \
    --run_name tacotron2_trainer \
    --working_directory <PATH_TO_WORKING_DIR> \
    --save_audio_gen <PATH_TO_SAVE_DIR> \
    --path_to_train_manifest <PATH_TO_TRAIN_METADATA> \
    --path_to_val_manifest <PATH_TO_TEST_METADATA> \
    --training_epochs 250 \
    --console_out_iters 5 \
    --wandb_log_iters 5 \
    --checkpoint_epochs 25 \
    --batch_size 64 \ # Batch size is per device (multiplied by num gpus)
    --learning_rate 0.001 \
    --min_learning_rate "1e-5" \
    --sampling_rate 22050 \
    --num_mels 80 \
    --n_fft 1024 \
    --window_size 1024 \
    --hop_size 256 \
    --min_db "-100" \
    --max_scaled_abs 4 \
    --fmin 0 \
    --fmax 8000 \
    --num_workers 32 \
    --log_wandb
```

You can see a full launch script in ```train_taco.sh``` that will let you mess with all the different options!

### Important Note

One thing to keep in mind is the scaling of the Mel Spectrograms. By default, we take our spectrograms and convert them to decibels scaling the values of our specrogram between roughly -100 to 0. I had some issues with convergence here even though this is the way the [NVIDIA/tacotron2](https://github.com/NVIDIA/tacotron2/tree/master) implementation had trained it (they did a simple log transform instead of decibel but same idea). So I followed closer to [kaituoxu/Tacotron2](https://github.com/kaituoxu/Tacotron2/tree/master) and scaled the values of the spectrogram between -4 and 4. Why 4? I don't think there is a reason, we could have done -5 to 5 or -1 to 1 and it would have all worked the same. But this makes sure our data is essentially zero centered and our neural networks will be happier with it!

You can see the details of this in the ```dataset.py``` file!

### Training Results

Generating Mel Spectrograms (especially with Teacher-Forcing) is not that hard. The tough part is learning the alignment between the text and the mel spectrograms. I plot here training results for a single sample for the first few epochs. Pay close attention to the alignment plot as we plot the attention weights of how much emphasis the model is placing on each character at each decoder step for the mel bins. This alignment is the most important part! 

<img src="https://github.com/priyammaz/PyTorch-Adventures/blob/main/src/visuals/taco_mel_gen.gif?raw=true" alt="drawing" width="400"/>


### Inference and Next Steps 

Go ahead and check out ```inference.ipynb``` to see how we can now use this model to generate audio, given a text prompt! If you listen to the audio though, you will notice something. It sounds robotic? This is because to generate audio from our Spectrograms we employ the Griffin Lim algorithm which is an iterative estimation of true audio, even though our phase information from the spectrograms is missing. 

To have better quality generations, we need something called a Neural Vocoder, that takes our spectrograms and converts them to much nicer sounding audio! That will be the next step when we train HIFIGAN!!