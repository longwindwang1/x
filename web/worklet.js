// Capture worklet: forwards raw Float32 input blocks to the main thread,
// which handles downsampling to 16 kHz PCM16 for the wire.
class CaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel && channel.length) {
      // Copy: the underlying buffer is reused by the audio engine.
      this.port.postMessage(new Float32Array(channel));
    }
    return true;
  }
}
registerProcessor('capture-processor', CaptureProcessor);
