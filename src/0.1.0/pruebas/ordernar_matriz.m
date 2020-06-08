function answw2
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
function solution=first(X, row, whichrowstoplace) %For a single root, loop over all leaves
solution=[];
for nr=1:length(X)-row %Try all possibilities to place the row + 1 in the correct position
    solution=next(X, row+1, nr, whichrowstoplace);
    if ~isnan(solution) %We got ourselves a solution, we can go back up a level
        return
    end
end
end

function solution=next(X, row, nr, whichrowstoplace)
firstrow=X(row,:);           %Take the row
onelocation=find(firstrow);  %And find the location of the ones
leaves=onelocation(ismember(onelocation,whichrowstoplace));  %We are only interested in the ones that we still need
if nr>length(leaves)         %You exhausted all possible tests, the row cannot be placed and therefore root must be bad
    solution=NaN;
else                         %Otherwise leaves(nr) is the next (candidate) solution
    solution=[leaves(nr) first(X, row, whichrowstoplace(whichrowstoplace~=leaves(nr)))];
end;
end