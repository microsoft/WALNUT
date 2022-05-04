### ATTENTION:
1. When set -1, it can run properly on my PC, but when set to 0, it throws errors.
2. The Toy scripts are amlt.yaml, but the local_rank cause the error when running on amulet. 


### ChangeLog:
0. Different Argument Parser.
   
   Current version add weak supervision dataset related arguments. For **clean_ratio**, please set 10 for imdb, yelp and 20 for agnews. 
   Please leave the default value for other not required arguments.  
``` python  
    parser.add_argument('--dataset', type=str, choices=['imdb', "agnews", "yelp", "political", "gossip"], default='imdb')
    parser.add_argument("--file_path", type=str, required=True, help="base directory of the data")
    parser.add_argument("--is_roberta", action="store_true", help="utilize the Roberta as the basic encoder or CNN")
    parser.add_argument("--weak_ratio", default=0.8, type=float, help="splittion of weak data and clean data")
    parser.add_argument("--clean_ratio", default=10, type=float, help="number of clean samples totally for imdb/agnews/yelp dataset")
    parser.add_argument("--n_high_cov", default=1, type=float, help="number of valid weak labeling functions for weak data")
```


1. Different Dataset implementation. Current version can load the imdb, agnews, yelp, gossipcop, political weak supervision dataset.
   
   The data is located at: https://drive.google.com/file/d/16plpxQuFx4dhmvZq1URHzGMZAz1CCYXa/view?usp=sharing 
2. New Evaluation Metrics. We will log the accuracy, f1 and confusion matrix. 


