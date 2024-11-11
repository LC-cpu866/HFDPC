% create a new server, called by main.m
classdef ServerNew
    properties % (Access = private) % private variables
        grid
        clG
    end
    properties
        n       % arnold algorithm：number of rounds
        a       % arnold algorithm：
        b       % arnold algorithm：
    end
    methods
        % ceate server objects
        function obj = ServerNew()
            obj.n = 5;
            obj.a = 5;
            obj.b = 6;
        end
        % Reduce the dimensions of the client. Receive information from the client and 
		% perform feature decomposition on the server to select appropriate feature vectors
        function users = DP_PCA(obj, users)
            numU = size(users,1);
            [count, sumCol] = users{1,1}.getInfo_DPPCA1();
            for i = 2:numU
                [aa, bb] = users{i,1}.getInfo_DPPCA1();
                count = count + aa;
                sumCol = sumCol + bb;
            end
            meanCol = sumCol./count;
            % When obtaining the three matrices H, B, and F for each client, consider using 
			% an image encryption algorithm (i.e. encrypting the matrices as images) 
            [encH, encB, encF, h] = users{1,1}.getInfo_DPPCA2(meanCol);
            H = decrypt_arnold(obj, encH, h, h);
            B = decrypt_arnold(obj, encB, 1, h);
            F = decrypt_arnold(obj, encF, h, 1);
            for i = 2:numU
                [encH, encB, encF, h] = users{i,1}.getInfo_DPPCA2(meanCol);
                H = H + decrypt_arnold(obj, encH, h, h);
                B = B + decrypt_arnold(obj, encB, 1, h);
                F = F + decrypt_arnold(obj, encF, h, 1);
            end
            % 4:
            numer = count*H - B'*B;
            deno = sqrt(count*F - B.^2) * sqrt(count*F - B.^2);
            R = numer./deno;
            % 5:
            [W, lambda] = eig(R);
            ev = abs((diag(lambda))');
            [ev, rhoev] = sort(ev, 'descend');
            W = W(:, rhoev);
            % 6:
            nn = length(ev);
            kk = zeros(1, nn);
            for i = 2:nn-1
                kk(i) = (ev(i)-ev(i-1))/(ev(i+1)-ev(i));
            end
            lin = find(kk == max(kk));
            meanEV = mean(ev(lin:end));
            stdEV = std(ev(lin:end));
            if ev(1) > meanEV+stdEV && lin > 2
                W = W(:,1:lin-1);
            else
                for cut = 1:nn
                    if sum(ev(1:cut))/sum(ev) > 0.99
                        break;
                    end
                end
                W = W(:,1:cut);
            end
			% When transmitting reduced dimensional feature vectors from the
			% server to the client, consider using encryption
            W = real(W); % avoid the complex numbers
            [h, w] = size(W);
            encW = obj.encrypt_arnold(W);
            for i = 1:numU
                users{i,1} = users{i,1}.DPPCA(encW, h, w);
            end
        end
        % use Arnold algorithm，encryption or chaotic
        function newX = encrypt_arnold(obj, oldX)
            [h, w] = size(oldX);
            if h>w
                max = h;
                B = ones(max, max-w)*125;
                img = cat(2, oldX, B);
            else
                max = w;
                B = ones(max-h, max)*125;
                img = cat(1, oldX, B);
            end
            [h, w] = size(img);
            N = h;
            newX = zeros(max, max);
            for i = 1:obj.n
                for y = 1:h
                    for x = 1:w
                        xx = mod((x-1)+obj.b*(y-1),N)+1;
                        yy = mod(obj.a*(x-1)+(obj.a*obj.b+1)*(y-1),N)+1;
                        newX(yy, xx) = img(y,x);
                    end
                end
                img = newX;
            end
        end
        % using Arnold algorithm，decryption or recovery
        function newX = decrypt_arnold(obj, oldX, s_h, s_w)
            [h, w] = size(oldX);
            newX = oldX;
            N = h;
            for i = 1:obj.n
                for y = 1:h
                    for x = 1:w
                        xx = mod((obj.a*obj.b+1)*(x-1)-obj.b*(y-1),N)+1;
                        yy = mod(-obj.a*(x-1)+(y-1),N)+1;
                        oldX(yy, xx) = newX(y, x);
                    end
                end
                newX = oldX;
            end
            newX = imcrop(oldX,[0, 0, s_w, s_h]);
        end
        % normalize data
        function users = Normalization(~, users)
            numU = size(users,1);
            [maxV, minV] = users{1,1}.getMaxMin();
            for i = 2:numU
                [maxI, minI] = users{i,1}.getMaxMin();
                maxV = max(maxV, maxI);
                minV = min(minV, minI);
            end
            for i = 1:numU
                users{i,1} = users{i,1}.Normal(minV, maxV);
            end
        end
        % calculate the divided grid points and the number of points within the grid
        function [obj, L] = Meshing(obj, users, L_theta, n, theta, NCLUST)
            numU = size(users,1);
            for L = 2:n
                gridT = users{1,1}.Partition(L);
                for i = 2:numU
                    gridU = users{i,1}.Partition(L);
                    gridT = [gridT; gridU];
                end
                [obj.grid, ~, count] = unique(gridT, 'rows');
                numG = sum(bsxfun(@eq, count, unique(count)'));
                if length(numG)/n >= L_theta
                    break;
                end
            end
            
            obj.clG = obj.SDC_DPC(numG, theta, L_theta, NCLUST);
        end
        % using SDC-DPC to label the grid
        function clG = SDC_DPC(obj, numG, theta, L_theta, NCLUST)
            Grid = obj.grid;
            clG = zeros(size(Grid,1),1)-1;
            K = ceil(sqrt(length(numG)));
            distM = pdist2(Grid, Grid);
            [sortM, ~] = sort(distM, 2);
            rhoG = exp(-1.*mean(sortM(:,2:K+1),2));
            numG = numG./max(numG);
            rhoG = rhoG./max(rhoG);
            rhoG = (1-L_theta)*numG' + L_theta*rhoG;
            
            [~, ordG] = sort(rhoG, 'descend');
            nneigh = zeros(size(Grid,1), 1);
            delta = zeros(size(Grid,1), 1);
            maxG = max(max(distM));
            for i = 2:size(Grid,1)
                delta(ordG(i)) = maxG;
                for j = 1:i-1
                    if distM(ordG(i), ordG(j)) < delta(ordG(i))
                        delta(ordG(i)) = distM(ordG(i), ordG(j));
                        nneigh(ordG(i)) = ordG(j);
                    end
                end
            end
            delta(ordG(1)) = maxG;
            
            R = rhoG.*delta;
            [~, ordR] = sort(R, 'descend');
            iclG = [];
            for i = 1:size(Grid,1)
                if clG(ordR(i)) > 0
                    continue;
                end
                if length(iclG) == NCLUST
                    break;
                end
                Ci = ordR(i);
                iclG = [iclG Ci];
                clG(ordR(i)) = length(iclG);
                neigh = find(distM(ordR(i),:)<2);
                while ~isempty(neigh)
                    next = neigh(1);
                    neigh(1) = [];
                    if clG(next) > 0
                        continue;
                    end
                    if rhoG(next) > theta*rhoG(Ci)
                        clG(next) = length(iclG);
                        item = find(distM(next,:)<2);
                        neigh = [neigh item];
                        % reduce judgment
                        neigh = unique(neigh);
                        neigh(clG(neigh)>0) = [];
                    else
                        clG(next) = length(iclG);
                    end
                end
            end
            
            for i = 1:length(ordG)
                if clG(ordG(i)) < 0
                    clG(ordG(i)) = clG(nneigh(ordG(i)));
                end
            end
        end
    end
end