%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script by Abhishek Kumar <abhishek.iitd16@gmail.com>
%
% It reads mnist_all.mat file and saves variables in multiple bin files
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Read and save bin files
clear
clc
load mnist_all.mat;

for file=1:2
	if file==1
		data_name='train';
	else
		data_name='test';
	end
	for digit=0:9
		fid = fopen(strcat(data_name,num2str(digit),'.bin'),'w');
		fwrite(fid,eval(strcat(data_name,num2str(digit))));
		fclose(fid);
	end
end


% Check if files are correct
clear
clc
load mnist_all.mat
for file=1:2
	if file==1
		data_name='train';
	else
		data_name='test';
	end
	for digit=0:9
		fid = fopen(strcat(data_name,num2str(digit),'.bin'),'r');
		A = reshape(fread(fid),[],784);
		fprintf('File: %s \t %d\n',strcat(data_name,num2str(digit)),isequal(A,eval(strcat(data_name,num2str(digit)))));
		fclose(fid);
	end
end

