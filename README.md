# Computer Vision example - Progressive Mirror Detection dataset insights and mirror detection
An example computer vision project displaying insights on the [Progressive Mirror Detection Dataset](https://jiaying.link/cvpr2020-pgd/).


## Get started
To get started, download the splited [Progressive Mirror Detection Dataset](https://jiaying.link/cvpr2020-pgd/) ZIP file and extract its content to a folder named `data` in the root directory of the project.


## Data insights
The notebook `data_insight.ipynb` loads the data from CSV file, displays graphs and descriptions of the data and the different columns.

1. Install the required packages
```console
pip install -r requirements.txt
```

2. Run the `data_insight.ipynb`


## Model training and validation
The scripts contained in the `scripts` contain the necessary code to train, validate and run the inference of the selected model.
It uses the local module `cvprogressivemirrordetection` containing the project-specific source code.

1. Install the `cvprogressivemirrordetection` module
```console
pip install .
```

2. Train the model
```console
python ./scripts/train.py
```
The scripts saves the model and the train/test metrics in the `output` folder.

3. Validate the model on new data
```console
python ./scripts/validation.py --file-path ./data/new_validation_data
```
