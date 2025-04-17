# Build stage
FROM node:16-alpine as build

# Set working directory
WORKDIR /app

# Copy package.json (don't require package-lock.json)
COPY package.json ./

# Create empty package-lock if not exists to support npm ci
RUN [ -f package-lock.json ] || echo '{}' > package-lock.json

# Install dependencies
RUN npm install

# Copy the rest of the application
COPY . .

# Set Docker-specific environment variables
ENV REACT_APP_ENVIRONMENT=docker
ENV REACT_APP_API_URL=''
ENV REACT_APP_ENABLE_SERVICE_WORKER=true

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built files from build stage
COPY --from=build /app/build /usr/share/nginx/html

# Add a health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 CMD wget -q --spider http://localhost:80

# Container runs nginx in foreground by default
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"] 