import * as handPoseDetection from '@tensorflow-models/hand-pose-detection';
import * as tf from '@tensorflow/tfjs-core';
// Register WebGL backend.
import '@tensorflow/tfjs-backend-webgl';
import * as tfn from '@tensorflow/tfjs-node';
import fs from 'fs';


const model = handPoseDetection.SupportedModels.MediaPipeHands;
const detectorConfig = {
  runtime: 'tfjs',
};

const detector = await handPoseDetection.createDetector(model, detectorConfig);


async function readImage() {
  const imageBuffer = await fs.readFileSync('test.jpg')
  console.log(imageBuffer);
  const md = tfn.node.decodeImage(imageBuffer, 3);
  console.log("MD: ", md);
  return md;
}


const estimationConfig = {};
const hands = await detector.estimateHands(await readImage(), estimationConfig);