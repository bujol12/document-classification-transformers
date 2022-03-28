This folder includes the original versions of the BEA2019, FCE and IMDB datasets. All of them provide documents that consist of sentences and tokens and are annotated on the token-level.

Each one of them follows a different structure, which is described in the folders of the original datasets.

We provide a script to convert these datasets into a unified JSON format, that follows the schema below:

```
{
    "documents" : [
        {
            "tokens": [["token1", "token2", ...], ...],
            "document_label": "label",
            "sentence_labels: ["label1", "label2", ...],
            "token_labels": [["label1", "label2", ...], ...],
            "meta": { }
        },
        ...
    ]
}
```

As in this task, we focus on binary prediction, we assume that the label==1 is positive and label==0 is negative. 

For Sentence- and document- labels are inferred in the case of GED datasets (FCE and BEA), with a positive label inferred iff there is at least one positive token (gramatically incorrect).

## BEA 2019

To convert, please run `./convert_bea2019.sh`

This will access the original BEA2019 dataset in the `original/` folder and convert it to our internal JSON representation. 
As BEA2019 only provides us with token-level labels, we only assign those, keeping other as null - that is in order to keep the original data structure. These nulls need to be handled by the data loader class of the model.
Additionally, this script will merge ABCN dev and train datasets to create the overall dev and train datasets.

As BEA2019 test dataset isn't public, we randomly sample 20\% of the train dataset as our internal `dev` dataset and re-label the original `dev` dataset as `test`.
