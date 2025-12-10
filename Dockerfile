FROM golang:1.24 AS build
ADD . /src
WORKDIR /src

RUN go get
RUN go test --cover -v ./...
RUN CGO_ENABLED=0 go build -v


FROM alpine:latest
# START FIX: Install timezone data package (tzdata) for TZ variable to work correctly
RUN apk add --no-cache tzdata
# END FIX
EXPOSE 8000
CMD [ "mlsolid" ]
COPY --from=build /src/mlsolid /usr/local/bin/mlsolid
RUN chmod +x /usr/local/bin/mlsolid
