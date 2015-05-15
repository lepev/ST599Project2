%%%%%Matlab code

%Reading the images from training set
train_path= 'C:/train/train';
cd(train_path); %Sets it as the work space
train_folder = dir(train_path);
train_labels = { train_folder.name };
marg=0; 
for n = 3:123
%removing the white margins of training set
folder_num = n;
location = strcat(train_path,'/',train_labels(folder_num));
location = char(location);
files = dir(fullfile(location,'*.jpg') );% import file names as characters
files = {files.name}; %% takes only the file names that are jpg
NumberOfImages = length(files);
for i = 1:NumberOfImages
image_num=i;
tmp_file = char(strcat(train_labels(folder_num),'/',files(image_num)));
im=imread(tmp_file);
im=double(im);

up=1;
down=size(im,1);
left=1;
right=size(im,2);

while sum(sum(255-im(up,:,:)))==0
up=up+1;
end

while sum(sum(255-im(down,:,:)))==0
down=down-1;
end

while sum(sum(255-im(:,left,:)))==0
left=left+1;
end

while sum(sum(255-im(:,right,:)))==0
right=right-1;
end

if up-marg>=1
up=up-marg;
end

if down+marg<=size(im,1)
down=down+marg;
end

if left-marg>=1
left=left-marg;
end

if right+marg<=size(im,2)
right=right+marg;
end

im=im(up:down,left:right,:);

im=uint8(im);
imwrite(im,tmp_file);

end
end

%splitting the training set to subtrain and validation set
trainset =imageSet(fullfile(train_path, char(train_labels{3})))
for i=4:123
trainset=[trainset,imageSet(fullfile(train_path, char(train_labels{i})))];
end
[trainset, validationSets] = partition(imgSets, 0.7, 'randomize');

%making the model on subtrain and validation sets
bag = bagOfFeatures(trainset);
categoryClassifier = trainImageCategoryClassifier(trainset, bag);

%extracting features from the whole training set
dd=0;
trainmat = NaN( 30335,502);
for i =1:121
for k=1:trainset(i).Count;
img = read(trainset(i), k);
dd=dd+1;
[r c]=size(img);
ws=sum(img(:) == 255)/(r*c);
ratio=c/r;
vec=[ws ratio];
trainmat(dd,:)=[encode(bag, img),vec];

end
end

%storing the features to use in R
csvwrite('C:/train/maintrainfeatures.csv',trainmat);

%removing the white margins of test set
NumberOfImages = length(files);
for i = 1:NumberOfImages
image_num=i;
tmp_file = char(files(image_num));
im=imread(tmp_file);
im=double(im);

up=1;
down=size(im,1);
left=1;
right=size(im,2);

while sum(sum(255-im(up,:,:)))==0
up=up+1;
end

while sum(sum(255-im(down,:,:)))==0
down=down-1;
end

while sum(sum(255-im(:,left,:)))==0
left=left+1;
end

while sum(sum(255-im(:,right,:)))==0
right=right-1;
end

if up-marg>=1
up=up-marg;
end

if down+marg<=size(im,1)
down=down+marg;
end

if left-marg>=1
left=left-marg;
end

if right+marg<=size(im,2)
right=right+marg;
end

im=im(up:down,left:right,:);

im=uint8(im);
imwrite(im,tmp_file);

end

%building the model on the whole training set
bag = bagOfFeatures(trainset);
categoryClassifier = trainImageCategoryClassifier(trainset, bag);

%extractin features from the test set
testcount=130400;
dd=0;
testmat = NaN(testcount,502);


for k=1:testcount;

img = read(testset,k);
dd=dd+1;
[r c]=size(img);
ws=sum(img(:) == 255)/(r*c);
ratio=c/r;
vec=[ws ratio];
testmat(dd,:)=[encode(bag, img),vec];

end

%storing the features to use in R
csvwrite('C:/train/testfeatures.csv',testmat);
  



######### R code
#loading the data
set.seed()
install.packages("randomForest")
library(randomForest)
install.packages("caret")
library(caret)

subtrain<-read.csv("subtrainfeatures.csv",header=T)
validation<-read.csv("validationfeatures.csv",header=T)
subtrainfeature<-subtrain[,1:502]
validfeature<-validation[,1:502]
subtrainresp<-subtrain[,503]
validresp<-validation[,503]

#find the best number of trees using subtrain and validation sets
modellist<-list()
myNtree  =  c(250,500,750,1000)
accuvec=c()
for ( i in 1:4){
  Sys.time()
  modellist[[i]] <- randomForest(y = subtrainresp, x = subtrainfeature,xtest = validfeature, ytest = validresp, ntree = myNtree[i],mtry=15,importance=TRUE, proximity=FALSE)
  cm.rf = confusionMatrix(data = (modellist[[i]]$test)$predicted, reference = validresp)
  accuvec[i]<-cm.rf$overall['Accuracy']
  print(i)
  Sys.time()
}

#finding the best variables to use
colnames(subtrainfeature)<-seq(1:502)
importancedf<-cbind(as.numeric(importance(modellist[[3]])[,122]),colnames(subtrainfeature)<-seq(1:502))
variablesorder<-(importancedf[order (importancedf[,1],decreasing=T),])[,2]

newmodellist<-list()
myNtree  =  750
accuvec2=c()
varnum<-seq(62,68,2)
for (i in 1:3){
  newmodellist[[i]] <- randomForest(y = subtrainresp, x = subtrainfeature[,variablesorder[1:varnum[i]]],xtest = validfeature[,variablesorder[1:varnum[i]]], ytest = validresp, ntree = 750,mtry=15,importance=TRUE, proximity=FALSE)
  cm.rf = confusionMatrix(data = (newmodellist[[i]]$test)$predicted, reference = validresp)
  accuvec2[i]<-cm.rf$overall['Accuracy']
  print(i)
  
}

#Now, making the model using the test and train set
rm(list=setdiff(ls(), c("maintrainresp","maintrainfeatures","testfeatures","variablesorder")))
gc()                        
finalmodel<-randomForest(y = maintrainresp, x = maintrainfeatures[,variablesorder[1:68]], ntree =750,mtry=15,importance=TRUE, proximity=FALSE)
testfeatures<-testfeatures[,variablesorder[1:68]]

#get the prediction matrix
predictions <- predict(finalmodel,testfeatures, type="prob")
saveRDS(predictions,file=paste(c("ranforpredictions"), collapse = " "),compress=TRUE)
write.csv(predictions, file = "ranfor_unscaled_prediction.csv")

#scaling the prediction matrix to avoid having log(0)
testmat<-predictions
addition<-c(0.01,0.001,0.0001,0.00001)
for (i in 1:4){
  scalevec<-rep.int(1+addition[i]*dim(testmat)[2],dim(testmat)[2])
  scaledmat<-t(t(testmat+addition[i])/(matrix(rep.int(scalevec,dim(testmat)[1]),ncol=dim(testmat)[1])))
  write.csv(scaledmat, file = paste(c("ranfor_scaled_prediction_",addition[i],".csv"), collapse = " "))
}

