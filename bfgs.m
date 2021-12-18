% To implement BFGS algorithm.
% Example:
% [xs,fs,k] = bfgs('f200','g200',[zeros(100,1);-ones(100,1)],1e-7);
function [xs,fs,k] = bfgs(D,fname,gname,mu,K,epsi)
format compact
format long
N1 = size(D,1);
muK = [mu K];
x0 = zeros(N1,K);
n = length(x0(:));
I = eye(n);
k = 1;
xk = x0(:);
Sk = I;
fk = feval(fname,xk,D,muK);
gk = feval(gname,xk,D,muK);
dk = -Sk*gk;
ak = bt_lsearch2019(xk,dk,fname,gname,D,muK);
dtk = ak*dk;
xk_new = xk + dtk;
fk_new = feval(fname,xk_new,D,muK);
dfk = abs(fk - fk_new);
er = max(dfk,norm(dtk));
while er >= epsi
      gk_new = feval(gname,xk_new,D,muK);
      gmk = gk_new - gk;
      D = dtk'*gmk;
      if D <= 0
         Sk = I;
      else
         sg = Sk*gmk;
         sw0 = (1+(gmk'*sg)/D)/D;
         sw1 = dtk*dtk';
         sw2 = sg*dtk';
         Sk = Sk + sw0*sw1 - (sw2'+sw2)/D;
      end
      fk = fk_new;
      gk = gk_new;
      xk = xk_new;
      dk = -Sk*gk;
      ak = bt_lsearch2019(xk,dk,fname,gname,D,muK);
      dtk = ak*dk;
      xk_new = xk + dtk;
      fk_new = feval(fname,xk_new,D,muK);
      dfk = abs(fk - fk_new);
      er = max(dfk,norm(dtk));
      k = k + 1;
end
disp('solution:')
xs = xk_new
disp('objective function at solution point:')
fs = feval(fname,xs)
format short
disp('number of iterations at convergence:')
k