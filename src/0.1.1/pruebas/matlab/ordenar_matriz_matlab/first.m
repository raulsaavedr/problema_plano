function solution = first(X, row, whichrowstoplace) 
    %For a single root, loop over all leaves
    solution=[];
    for nr=1:length(X)-row %Try all possibilities to place the row + 1 in the correct position
        disp('Adentro de first ciclo principal');
        disp('nr:');disp(nr);
        disp('length(X)'); disp(length(X));
        disp('row:'); disp(row);
        disp('whichrowstoplace:'); disp(whichrowstoplace);
        solution=next(X, row+1, nr, whichrowstoplace);
        if ~isnan(solution) %We got ourselves a solution, we can go back up a level
            return
        end
    end
end
