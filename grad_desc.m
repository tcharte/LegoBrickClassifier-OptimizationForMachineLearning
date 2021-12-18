% To implement the gradient descent algorithm.
% Example: [xs,fs,k] = grad_desc('f_rosen','g_rosen',[0; 2],1e-9);
function [xs,fs,k] = grad_desc(D,fname,gname,mu,K,epsi)
format compact
format long
N1 = size(D,1);
muK = [mu K];
x0 = zeros(N1,K);
k = 1;
xk = x0;
gk = feval(gname,xk,D,muK);
dk = -gk;
ak = bt_lsearch2019(xk,dk,fname,gname,D,muK);
adk = ak*dk;
er = norm(adk);
while er >= epsi
  xk = xk + adk;
  gk = feval(gname,xk,D,muK);
  dk = -gk;
  ak = bt_lsearch2019(xk,dk,fname,gname,D,muK);
  adk = ak*dk;
  er = norm(adk);
  k = k + 1;
end
disp('solution:')
xs = xk + adk
disp('objective function at solution point:')
fs = feval(fname,xs,D,muK)
format short
disp('number of iterations performed:')
k