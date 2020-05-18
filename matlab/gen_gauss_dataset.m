
function [] = gen_gauss_dataset ()
	D = [];
	L = [];
	for i = 1:10
		A = mvnrnd( i, 0.5, 100*3072); 
		B = i * ones( 100, 1); 
		A = reshape( A, 100, 3072); 
		D = [D; A]; 
		L = [L; B];
	end
	dlmwrite( 'gauss-dataset.txt', D); 
	dlmwrite( 'gauss-labels.txt', L ); 
end
