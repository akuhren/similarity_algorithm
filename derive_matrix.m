function res = derive_matrix(aspect, fs)
% all samples with all features
M_full = table2cell(readtable('feature_data.txt'));

% matrix with each column corresponding to a feature and each row
% corresponding to an audio sample
M_feat = cell2mat(M_full(:,2:end));

% standardize feature matrix (should be done earlier in the process)
for i = 1:length(M_feat(1,:))
    col = M_feat(:,i)';
    col(isnan(col)) = 0;
    s2 = std(col);
    mu = mean(col);
    new = arrayfun(@(x) (x-mu)/s2, col);
    M_feat(:,i) = new';
end

% create mapping from hashes to indices for efficiency
map = containers.Map();
for i=1:length(M_full(:,1))
    map(char(M_full(i,1))) = i;
end

% by default, set aspect to 'punch'
if ~exist('aspect', 'var')
    aspect = 'punch';
end

% by default, consider the first 10 features
if ~exist('fs', 'var')
    fs = 1:10;
end

% standardize by 2 to avoid Inf cost-results
sigma2 = 2;


% data from feature tests
url = strcat('testdata_', aspect, '.csv');
T = table2cell(readtable(url));
T = T(1:10,:); % TO BE DELETED


% encode table so that each test corresponds to two rows:
% +----+----+----+----+
% | x1 | x2 | x1 | x3 |
% +----+----+----+----+
% | x1 | x2 | x2 | x3 |
% +----+----+----+----+
% where x3 is the odd sample
T_enc = data_enc(T);
function res = data_enc(M)
    res = cell(1, 4);
    k = 1;
    for i = 1:length(M(:,1))
        row = M(i,:);
        odd = cell2mat(M(i,4));
        if(odd == 4)
            % for now, ignore "cant' tell"-answers
            continue;
        end
        
        others = row(1:3 ~= odd);
        x1 = others(1);
        x2 = others(2);
        x3 = row(odd);
        
        res(k:(k+1),1) = x1;
        res(k:(k+1),2) = x2;
        res(k:(k+1),4) = x3;

        res(k,3) = x1;
        res(k+1,3) = x2;
        k = k+2;
    end 
end


% set K to identity matrix
K = eye(length(fs));
[maxJ, err, index] = checkgrad(@optim, unwrap_dia(K), 1e-6*(1+norm(K)), T_enc);

% should be close to zero
res = err(1)+2*err(2);

% wrap (1 x n^2)-vector to (n x n)-matrix
function out = wrap(in)
    s = sqrt(length(in));
    out = zeros(s, s);
    for i = 1:s
        for j = 1:s
            out(i,j) = in((i-1)*s+j);
        end
    end
end

% unwrap (n x n)-matrix to (1 x n^2)-vector
function out = unwrap(in)
    out = zeros(1, length(in)^2);
    for i = 1:length(in)
        for j = 1:length(in)
            out((i-1)*length(in)+j) = K(i,j);
        end
    end
end

% construct a diagonal matrix provided the diagonal values in a vector
function out = wrap_dia(in)
    s = length(in);
    out = zeros(s,s);
    for i = 1:s
        out(i,i) = in(i);
    end
end

% extract diagonal from matrix
function out = unwrap_dia(in)
    s = length(in);
    out = zeros(1,s);
    for i = 1:s
        out(i) = in(i,i);
    end
end

% cost- and gradient-function we want to minimize
function [cost, grad] = optim(unwrapped, data)
    K = wrap_dia(unwrapped);
    nrows = length(data(:,1));
    
    cost = 0;
    for row = data'
        d1 = d(row(1), row(2), K); % short distance
        d2 = d(row(3), row(4), K); % long distance
        log_prob = log(cdf('norm', (d2-d1)/sigma2));
        cost = cost - log_prob;
    end
    
    % stop if gradient is not required
    if nargin < 2
        return
    end

    % negative sum of all test- gradients
    grad = -sum(arrayfun(@(i) test_grad(data(i,:),K), 1:nrows));
    
    % checkgrad seems to want the gradient as a vector with same
    % length as the unwrapped K, so for now we just replicate it
    grad = repmat(grad,1, length(unwrapped));
end

function grad = test_grad(row, K)
    d1 = d(row(1), row(2), K); % short distance
    d2 = d(row(3), row(4), K); % long distance

    p_grad = pdf('norm', (d2-d1)/sigma2)/cdf('norm', (d2-d1)/sigma2);
    d_grad1 = dist_grad(row(1),row(2), K);
    d_grad2 = dist_grad(row(3),row(4), K);

    grad = p_grad/sigma2*(d_grad2-d_grad1);
end

function grad = dist_grad(h1, h2, K)
    % if K = L*L'
    mat_grad = K' + K;
    % if K is diagonal
    mat_grad = eye(length(K));

    x1 = M_feat(map(char(h1)),fs)';
    x2 = M_feat(map(char(h2)),fs)';

    grad = (x2-x1)'*mat_grad*(x2-x1);
end

function dist = d(h1, h2, K)
   x1 = M_feat(map(char(h1)), fs)'; % feature vector 1
   x2 = M_feat(map(char(h2)), fs)'; % feature vector 2
   
   % For now, replace NaN's with 0
   x1(isnan(x1)) = 0;
   x2(isnan(x2)) = 0;
   
   dist = (x1-x2)'*K*(x1-x2); 
end

end