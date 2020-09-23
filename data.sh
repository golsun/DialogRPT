# step 0. create the data folder

mkdir "data/bz2"

# Step 1. Download raw data from a third party dump: https://files.pushshift.io/reddit

# download comments for year 2011
wget https://files.pushshift.io/reddit/comments/RC_2011-01.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/comments/RC_2011-02.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/comments/RC_2011-03.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/comments/RC_2011-04.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/comments/RC_2011-05.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/comments/RC_2011-06.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/comments/RC_2011-07.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/comments/RC_2011-08.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/comments/RC_2011-09.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/comments/RC_2011-10.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/comments/RC_2011-11.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/comments/RC_2011-12.bz2 -P data/bz2

# download comments for year 2012
wget https://files.pushshift.io/reddit/comments/RC_2012-01.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/comments/RC_2012-02.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/comments/RC_2012-03.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/comments/RC_2012-04.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/comments/RC_2012-05.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/comments/RC_2012-06.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/comments/RC_2012-07.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/comments/RC_2012-08.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/comments/RC_2012-09.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/comments/RC_2012-10.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/comments/RC_2012-11.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/comments/RC_2012-12.bz2 -P data/bz2

# download submissions for year 2011
wget https://files.pushshift.io/reddit/submissions/RS_2011-01.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/submissions/RS_2011-02.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/submissions/RS_2011-03.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/submissions/RS_2011-04.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/submissions/RS_2011-05.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/submissions/RS_2011-06.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/submissions/RS_2011-07.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/submissions/RS_2011-08.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/submissions/RS_2011-09.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/submissions/RS_2011-10.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/submissions/RS_2011-11.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/submissions/RS_2011-12.bz2 -P data/bz2

# download submissions for year 2011
wget https://files.pushshift.io/reddit/submissions/RS_2012-01.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/submissions/RS_2012-02.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/submissions/RS_2012-03.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/submissions/RS_2012-04.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/submissions/RS_2012-05.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/submissions/RS_2012-06.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/submissions/RS_2012-07.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/submissions/RS_2012-08.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/submissions/RS_2012-09.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/submissions/RS_2012-10.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/submissions/RS_2012-11.bz2 -P data/bz2
wget https://files.pushshift.io/reddit/submissions/RS_2012-12.bz2 -P data/bz2

# Step 2. Read the `.bz2` files and group items from the same subreddit 

python src/data.py bz2 2011
python src/data.py bz2 2012

# Step 3. extract basic attributes and dialog trees.

python src/data.py basic 2011
python src/data.py basic 2012

# Step 4. Build training and testing data for different feedback signals. 

python src/data.py updown 2011 --year_to=2012
python src/data.py depth 2011 --year_to=2012
python src/data.py width 2011 --year_to=2012