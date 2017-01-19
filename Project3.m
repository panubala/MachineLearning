h5disp('train.h5');
X = h5read('train.h5','/train/block0_values');
X = transpose(X);
whos X;
Y = h5read('train.h5','/train/block1_values');
Y2 = [VarName2,VarName3, VarName4, VarName5,VarName6];
whos Y;
X1 = transpose(X(1:1500,:));
Y1 = transpose(Y2(1:1500,:));
whos X1;
whos Y1;

net = patternnet(10);
view(net);
net.divideParam.trainRatio = .7;
net.divideParam.valRatio = .15;
net.divideParam.testRatio = .15;

[net,tr]=train(net,X1, Y1);

Xtest = h5read('test.h5', '/test/block0_values');
whos Xtest;

output = net(Xtest);
output = transpose(output);
whos otput;
Id = transpose(linspace(45324,53460,8137));
whos Id;
C = [Id,output];
whos C;
csvwrite('submission3_3.csv',C);

