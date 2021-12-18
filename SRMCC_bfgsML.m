% Softmax regression for multi-class classification (SRMCC).
% The minimization is carried out by ML-BFGS.
% Inputs:
% Dtr: (N+1) x P train data matrix of K classes of samples. 
%      Its last row are class labels between 1 and K.
% fname: regularized softmax cost function defined by Eq. (3.6).
% gname: gradient of the cost function given by Eqs.(3.8a) and (3.8b).
% mu: regularization parameter \mu.
% K: number of classes in the data sets.
% iter: number of iterations to be performed.
% Outputs:
% Ws: optimized weight-and-bias matrix of size (N+1) by K.
% f: profile of softmax cost function values.
% Prepared by W.-S. Lu, University of Victoria.
% Last modofied: June 28, 2020.
function [Ws,f,k] = SRMCC_bfgsML(Dtr,fname,gname,mu,K,epsi)
N1 = size(Dtr,1);
muK = [mu K];
W0 = zeros(N1,K);
k = 1;
xk = W0(:);
fk = feval(fname,xk,Dtr,muK);
f = fk;
fprintf('iter %1i: loss = %8.2e\n',k-1,fk)
gk = feval(gname,xk,Dtr,muK);
dk = -gk;
ak = bt_lsearch2019(xk,dk,fname,gname,Dtr,muK);
dtk = -ak*gk;
xk_new = xk + dtk;
fk_new = feval(fname,xk_new,Dtr,muK);
dfk = abs(fk - fk_new);
er = max(dfk,norm(dtk));
fk = fk_new;
f = [f; fk];
fprintf('iter %1i: loss = %8.2e\n',k,fk)
while er >= epsi
  gk_new = feval(gname,xk_new,Dtr,muK);
  gmk = gk_new - gk;
  gk = gk_new;
  rk = 1/(dtk'*gmk);
  if rk <= 0
     dk = -gk;
  else
     tk = dtk'*gk;
     qk = gk - (rk*tk)*gmk;
     bk = rk*(gmk'*qk - tk);
     dk = bk*dtk - qk;
  end
  xk = xk_new;
  ak = bt_lsearch2019(xk,dk,fname,gname,Dtr,muK);
  dtk = ak*dk;
  xk_new = xk + dtk;
  fk_new = feval(fname,xk_new,D,muK);
  dfk = abs(fk - fk_new);
  er = max(dfk,norm(dtk));
  fk = fk_new;
  f = [f; fk];
  k = k + 1;
  fprintf('iter %1i: loss = %8.2e\n',k,fk)
end
Ws = reshape(xk_new,N1,K);
fprintf('final loss = %8.2e\n',fk)