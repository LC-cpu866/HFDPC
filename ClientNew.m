% create a new client class and call it by main.m
classdef ClientNew
    properties %(Access = private) % set private variables
        data
    end
    properties
        labels  % used to store the label of data, compared with real result
        L       % the size of grid
        cl      % clustering result
        n       % arnold algorithm：number of rounds
        a       % arnold algorithm：
        b       % arnold algorithm：
    end
    methods
        % 1)initalization
        function obj = ClientNew(X, label)
            obj.data = X;
            obj.labels = label;
            obj.n = 5;
            obj.a = 5;
            obj.b = 6;
        end
        % 2)data normalization: first, obtain the maximum and minimum features of each client,
		%                        and then normalize them
        function [maxV, minV] = getMaxMin(obj)
            maxV = max(obj.data);
            minV = min(obj.data);
        end
        function obj = Normal(obj, minV, maxV)
            X = (obj.data - minV)./(maxV - minV);
            X(isnan(X)) = 0;
            obj.data = X;
        end
        % 3)distributed data dimensionality reduction
        function [count, sumCol] = getInfo_DPPCA1(obj)
            % Input：obj
            % Output：count   :data size
            %         sumCol  :the sum of each column of data
            X = obj.data;
            count = size(X,1);
            sumCol = sum(X);
        end
        function [encH, encB, encF, h] = getInfo_DPPCA2(obj, meanCol)
            % Input：obj
            % Output  H       :the square matrix
            %         B       :the linear sum vector
            %         F       :the square sum vector
            X = obj.data;
            meanU = ones(size(X, 1),1) * meanCol;
            user_data = X - meanU;
            H = user_data'*user_data;
            B = sum(user_data);
            F = diag(H);
            h = length(F); % record the height and width of H, B, and F. H(h*h),B(1*h),F(h*1);
            encH = obj.encrypt_arnold(H);
            encB = obj.encrypt_arnold(B);
            encF = obj.encrypt_arnold(F);
        end
        function obj = DPPCA(obj, encW, h, w)
            % reduce the dimensions of client data by the matrix W obtained through feature decomposition
            % receive the feature vector matrix from the server and decrypt it
            W = obj.decrypt_arnold(encW, h, w);
            obj.data = real(obj.data*W);
        end
        % divide the data into grids based on the given grid size
        function gridU = Partition(obj, L)
            X = obj.data;
            gridU = ceil(L.*X);
        end
        % assign data labels based on the input grid points and corresponding grid point labels
        function obj = Assign(obj, Grid, clG, L)
            X = ceil(L.*obj.data);
            distXG = pdist2(X, Grid);
            [~, ordXG] = sort(distXG, 2);
            obj.cl = clG(ordXG(:,1));
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
        % use Arnold algorithm，decryption or recovery
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
    end
end