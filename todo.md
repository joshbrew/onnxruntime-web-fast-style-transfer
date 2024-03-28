
- Retrain Small Net with the smaller channel counts
- Retrain Small Efficient Net to test different in pixel shuffle layer swap difference
- Retrain the large net we had originally
- Retrain the large efficient net since we forgot a group conv in it and also we had missing padding, anyway it probably wont be good

- Try super resolution alternative e.g. to do a 3x upsample, not sure it will work with dynamic res