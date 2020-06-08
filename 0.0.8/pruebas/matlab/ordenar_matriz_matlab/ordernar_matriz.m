function ordenar_matriz
    n=5 ; %Generate a matrix that fulfills the requirements
    xoriginal=diag(ones(n,1));
    y= rand(n,n);
    xoriginal(y<(n/2-1)/(n-1))=1; %so that on average as much 0's as 1's (so sum(x(:))/n^2 = 0.5 on average)
    x = xoriginal(randperm(n),:); %And permute it so that we have a challenge
      % Alternatively use x=round(rand(n,n)) for a random 1/0 matrix with no guaranteed solution
      % Now put the rows with fewest 1's on top, as it is best to start with these
      [~,isorted]=sort(sum(x,2));  % is a column vector containing the sum of each row.
      x=x(isorted,:);
      %And now find the solution
      per=first(x,0,1:n);
      %Output the solution
      disp('This is the initial solution'); xoriginal
      disp('Shuffled version'); x
      if ~isnan(per)
          disp('This is the solution found'); xsol(per,:)=x
          disp('Order of the rows: '); per
          assert(all(diag(xsol)==1)) %Sanity test
      else
          disp('No solution found !');
          return
      end;
end