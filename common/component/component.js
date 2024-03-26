import { isElectron, isWebNN } from "../utils.js";

const webnnlogo = () => {
  const nnlogo = `
            <svg
            viewBox="2.6957588027748756 5.85526315789474 112.52568466533302 24.310003913380942"
            width="562.65"
            height="121.55"
            class="webnn"
            xmlns:xlink="http://www.w3.org/1999/xlink"
            xmlns="http://www.w3.org/2000/svg"
        >
            <defs>
            <path
                d="M64.79 7.02C65.78 7.02 66.58 7.82 66.58 8.81C66.58 10.97 66.58 15.65 66.58 17.8C66.58 18.79 65.78 19.59 64.79 19.59C59.8 19.59 46.64 19.59 41.66 19.59C40.67 19.59 39.86 18.79 39.86 17.8C39.86 15.65 39.86 10.97 39.86 8.81C39.86 7.82 40.67 7.02 41.66 7.02C46.64 7.02 59.8 7.02 64.79 7.02Z"
                id="aUJMA4bMI"
            ></path>
            <text
                id="c5jvJP04B"
                x="63.6"
                y="121.55"
                font-size="10"
                font-family="Poppins"
                font-weight="100"
                alignment-baseline="before-edge"
                transform="matrix(1 0 0 1 -66.79615093164342 -104.194166492847)"
                style="line-height: 100%"
                xml:space="preserve"
                dominant-baseline="text-before-edge"
            >
                <tspan
                x="61"
                dy="0em"
                alignment-baseline="before-edge"
                dominant-baseline="text-before-edge"
                text-anchor="start"
                >
                neural network
                </tspan>
            </text>
            <text
                id="a1vYTZuYgt"
                x="138.5"
                y="89"
                font-size="10"
                font-family="Poppins"
                alignment-baseline="before-edge"
                transform="matrix(1 0 0 1 -136.17057369513643 -82.54473073858966)"
                style="line-height: 100%"
                xml:space="preserve"
                dominant-baseline="text-before-edge"
            >
                <tspan
                x="133.5"
                dy="0em"
                alignment-baseline="before-edge"
                dominant-baseline="text-before-edge"
                text-anchor="start"
                >
                web
                </tspan>
            </text>
            <path
                d="M29.98 27.82C29.98 29.12 28.93 30.17 27.64 30.17C26.35 30.17 25.3 29.12 25.3 27.82C25.3 26.53 26.35 25.48 27.64 25.48C28.93 25.48 29.98 26.53 29.98 27.82Z"
                id="m1AZcqfjwA"
            ></path>
            <path
                d="M22.48 13.75L22.91 14.38L23.06 15.15L22.91 15.92L22.48 16.55L21.85 16.98L21.08 17.13L20.31 16.98L19.68 16.55L19.26 15.92L19.1 15.15L19.26 14.38L19.68 13.75L20.31 13.33L21.08 13.17L21.85 13.33L22.48 13.75ZM20.29 14.36L20.04 14.72L19.96 15.16L20.04 15.6L20.29 15.96L20.65 16.2L21.09 16.29L21.53 16.2L21.89 15.96L22.13 15.6L22.22 15.16L22.13 14.72L21.89 14.36L21.53 14.11L21.09 14.03L20.65 14.11L20.29 14.36Z"
                id="a99wrsIm5"
            ></path>
            <path
                d="M21.08 21.31C21.08 22.61 20.03 23.65 18.74 23.65C17.45 23.65 16.4 22.61 16.4 21.31C16.4 20.02 17.45 18.97 18.74 18.97C20.03 18.97 21.08 20.02 21.08 21.31Z"
                id="b2fOGzz6El"
            ></path>
            <path
                d="M36.24 20.16L36.75 20.9L36.93 21.82L36.75 22.73L36.24 23.47L35.5 23.97L34.59 24.16L33.68 23.97L32.93 23.47L32.43 22.73L32.25 21.82L32.43 20.9L32.93 20.16L33.68 19.66L34.59 19.47L35.5 19.66L36.24 20.16ZM33.65 20.88L33.36 21.3L33.26 21.82L33.36 22.34L33.65 22.77L34.08 23.05L34.6 23.16L35.12 23.05L35.54 22.77L35.83 22.34L35.93 21.82L35.83 21.3L35.54 20.88L35.12 20.59L34.6 20.48L34.08 20.59L33.65 20.88Z"
                id="hfanhELPS"
            ></path>
            <path
                d="M17.01 11.16C17.01 12.46 15.96 13.5 14.67 13.5C13.38 13.5 12.33 12.46 12.33 11.16C12.33 9.87 13.38 8.82 14.67 8.82C15.96 8.82 17.01 9.87 17.01 11.16Z"
                id="g3EBuH3JyM"
            ></path>
            <path
                d="M24.08 7.44L24.39 7.9L24.5 8.46L24.39 9.02L24.08 9.48L23.62 9.79L23.06 9.9L22.5 9.79L22.04 9.48L21.73 9.02L21.62 8.46L21.73 7.9L22.04 7.44L22.5 7.13L23.06 7.02L23.62 7.13L24.08 7.44ZM22.48 7.88L22.31 8.15L22.24 8.47L22.31 8.79L22.48 9.05L22.75 9.22L23.07 9.29L23.39 9.22L23.65 9.05L23.82 8.79L23.89 8.47L23.82 8.15L23.65 7.88L23.39 7.71L23.07 7.64L22.75 7.71L22.48 7.88Z"
                id="a3SqsBDOg"
            ></path>
            <path
                d="M28.48 20.8L28.79 21.25L28.9 21.82L28.79 22.38L28.48 22.83L28.02 23.14L27.46 23.26L26.9 23.14L26.44 22.83L26.13 22.38L26.02 21.82L26.13 21.25L26.44 20.8L26.9 20.49L27.46 20.37L28.02 20.49L28.48 20.8ZM26.88 21.24L26.71 21.5L26.64 21.82L26.71 22.14L26.88 22.4L27.14 22.58L27.46 22.64L27.78 22.58L28.05 22.4L28.22 22.14L28.29 21.82L28.22 21.5L28.05 21.24L27.78 21.06L27.46 21L27.14 21.06L26.88 21.24Z"
                id="i15BKTlpWe"
            ></path>
            <path
                d="M9.79 12.15L10.1 12.61L10.21 13.17L10.1 13.73L9.79 14.19L9.33 14.5L8.77 14.61L8.21 14.5L7.75 14.19L7.44 13.73L7.33 13.17L7.44 12.61L7.75 12.15L8.21 11.84L8.77 11.73L9.33 11.84L9.79 12.15ZM8.19 12.59L8.02 12.85L7.95 13.17L8.02 13.49L8.19 13.76L8.45 13.93L8.77 14L9.09 13.93L9.36 13.76L9.53 13.49L9.6 13.17L9.53 12.85L9.36 12.59L9.09 12.42L8.77 12.35L8.45 12.42L8.19 12.59Z"
                id="a7ILsoRCN"
            ></path>
            <path
                d="M34.24 25.9L34.55 26.36L34.66 26.92L34.55 27.48L34.24 27.94L33.78 28.25L33.22 28.36L32.66 28.25L32.2 27.94L31.9 27.48L31.78 26.92L31.9 26.36L32.2 25.9L32.66 25.6L33.22 25.48L33.78 25.6L34.24 25.9ZM32.65 26.35L32.47 26.61L32.4 26.93L32.47 27.25L32.65 27.51L32.91 27.69L33.23 27.75L33.55 27.69L33.81 27.51L33.99 27.25L34.05 26.93L33.99 26.61L33.81 26.35L33.55 26.17L33.23 26.1L32.91 26.17L32.65 26.35Z"
                id="e1SP5zBvIe"
            ></path>
            <path
                d="M31.06 13.17C31.06 14.46 30.01 15.51 28.72 15.51C27.43 15.51 26.38 14.46 26.38 13.17C26.38 11.88 27.43 10.83 28.72 10.83C30.01 10.83 31.06 11.88 31.06 13.17Z"
                id="bfBySayYv"
            ></path>
            <path
                d="M21.74 25.9L22.05 26.36L22.16 26.92L22.05 27.48L21.74 27.94L21.28 28.25L20.72 28.36L20.16 28.25L19.7 27.94L19.39 27.48L19.28 26.92L19.39 26.36L19.7 25.9L20.16 25.6L20.72 25.48L21.28 25.6L21.74 25.9ZM20.14 26.35L19.97 26.61L19.9 26.93L19.97 27.25L20.14 27.51L20.4 27.69L20.72 27.75L21.04 27.69L21.31 27.51L21.48 27.25L21.55 26.93L21.48 26.61L21.31 26.35L21.04 26.17L20.72 26.1L20.4 26.17L20.14 26.35Z"
                id="b1cdUukPld"
            ></path>
            <path
                d="M14.97 16.11L15.28 16.57L15.39 17.13L15.28 17.69L14.97 18.15L14.51 18.46L13.95 18.57L13.39 18.46L12.93 18.15L12.62 17.69L12.51 17.13L12.62 16.57L12.93 16.11L13.39 15.8L13.95 15.69L14.51 15.8L14.97 16.11ZM13.37 16.55L13.2 16.82L13.13 17.14L13.2 17.46L13.37 17.72L13.64 17.89L13.96 17.96L14.28 17.89L14.54 17.72L14.71 17.46L14.78 17.14L14.71 16.82L14.54 16.55L14.28 16.38L13.96 16.31L13.64 16.38L13.37 16.55Z"
                id="eK2gZrDmt"
            ></path>
            <path
                d="M7.38 17.85C7.38 19.15 6.33 20.19 5.04 20.19C3.74 20.19 2.7 19.15 2.7 17.85C2.7 16.56 3.74 15.51 5.04 15.51C6.33 15.51 7.38 16.56 7.38 17.85Z"
                id="d7YHIEVZ2"
            ></path>
            <path d="M7.85 14.43L5.31 17.98" id="a17Or6DtLG"></path>
            <path d="M13.44 16.2L9.48 13.7" id="j4QWBAJZw"></path>
            <path d="M11.46 23.78L13.44 18.59" id="bhjRTNplq"></path>
            <path d="M16.57 20.69L7.19 17.98" id="cUv46xKpz"></path>
            <path d="" id="a86zQgXCB"></path>
            <path d="M19.81 15.74L14.79 16.81" id="cf4JQTqTn"></path>
            <path d="M26.24 13.7L23.1 14.43" id="a2mhrsE0Kx"></path>
            <path d="" id="bmGTT9MeB"></path>
            <path d="M21.75 8.99L16.97 10.37" id="c5ckw7gUg"></path>
            <path d="M21.75 13.13L22.49 9.68" id="bNq0DnU5M"></path>
            <path d="M27.39 20.69L27.87 15.74" id="b1AsRfq0o"></path>
            <path d="M21.75 26.3L26.24 22.63" id="a5Fonrsblc"></path>
            <path d="M26.24 21.19L22.12 16.81" id="a6xLjyPrwa"></path>
            <path d="M32.76 22.02L28.9 22.02" id="feqWMbnGx"></path>
            <path d="M17.88 19.33L15.59 13.13" id="b877fx0LIT"></path>
            <path d="M25.86 26.92L20.78 22.63" id="aHbsBwJX"></path>
            <path d="M32.04 27.54L30.11 28.46" id="b7wp4SAvn"></path>
            <path d="" id="c1lg9Iyd5S"></path>
            <path d="" id="b531Gp81NK"></path>
            <path d="M33.52 26.3L34.39 23.78" id="c4gBuIj7W"></path>
            <path d="M35.36 30.17L33.98 28" id="g75dLLAPY"></path>
            <path d="M33.96 19.91L30.46 14.95" id="b9EtFKjjmd"></path>
            </defs>
            <g>
            <g>
                <g>
                <use
                    xlink:href="#aUJMA4bMI"
                    opacity="1"
                    fill="rgba(255, 255, 255, 1.0)"
                    fill-opacity="0.4"
                ></use>
                </g>
                <g id="c5WataF5oH">
                <use
                    xlink:href="#c5jvJP04B"
                    opacity="1"
                    fill="rgba(255, 255, 255, 1.0)"
                    fill-opacity="1"
                ></use>
                </g>
                <g id="flQxJVdR">
                <use
                    xlink:href="#a1vYTZuYgt"
                    opacity="1"
                    fill="rgba(255, 255, 255, 1.0)"
                    fill-opacity="1"
                ></use>
                </g>
                <g>
                <g>
                    <use
                    xlink:href="#m1AZcqfjwA"
                    opacity="1"
                    fill="rgba(255, 255, 255, 1.0)"
                    fill-opacity="1"
                    ></use>
                </g>
                <g>
                    <use
                    xlink:href="#a99wrsIm5"
                    opacity="1"
                    fill="rgba(255, 255, 255, 1.0)"
                    fill-opacity="1"
                    ></use>
                </g>
                <g>
                    <use
                    xlink:href="#b2fOGzz6El"
                    opacity="1"
                    fill="rgba(255, 255, 255, 1.0)"
                    fill-opacity="1"
                    ></use>
                </g>
                <g>
                    <use
                    xlink:href="#hfanhELPS"
                    opacity="1"
                    fill="rgba(255, 255, 255, 1.0)"
                    fill-opacity="1"
                    ></use>
                </g>
                <g>
                    <use
                    xlink:href="#g3EBuH3JyM"
                    opacity="1"
                    fill="rgba(255, 255, 255, 1.0)"
                    fill-opacity="1"
                    ></use>
                </g>
                <g>
                    <use
                    xlink:href="#a3SqsBDOg"
                    opacity="1"
                    fill="rgba(255, 255, 255, 1.0)"
                    fill-opacity="1"
                    ></use>
                </g>
                <g>
                    <use
                    xlink:href="#i15BKTlpWe"
                    opacity="1"
                    fill="rgba(255, 255, 255, 1.0)"
                    fill-opacity="1"
                    ></use>
                </g>
                <g>
                    <use
                    xlink:href="#a7ILsoRCN"
                    opacity="1"
                    fill="rgba(255, 255, 255, 1.0)"
                    fill-opacity="1"
                    ></use>
                </g>
                <g>
                    <use
                    xlink:href="#e1SP5zBvIe"
                    opacity="1"
                    fill="rgba(255, 255, 255, 1.0)"
                    fill-opacity="1"
                    ></use>
                </g>
                <g>
                    <use
                    xlink:href="#bfBySayYv"
                    opacity="1"
                    fill="rgba(255, 255, 255, 1.0)"
                    fill-opacity="1"
                    ></use>
                </g>
                <g>
                    <use
                    xlink:href="#b1cdUukPld"
                    opacity="1"
                    fill="rgba(255, 255, 255, 1.0)"
                    fill-opacity="1"
                    ></use>
                </g>
                <g>
                    <use
                    xlink:href="#eK2gZrDmt"
                    opacity="1"
                    fill="rgba(255, 255, 255, 1.0)"
                    fill-opacity="1"
                    ></use>
                </g>
                <g>
                    <use
                    xlink:href="#d7YHIEVZ2"
                    opacity="1"
                    fill="rgba(255, 255, 255, 1.0)"
                    fill-opacity="1"
                    ></use>
                </g>
                <g>
                    <g>
                    <use
                        xlink:href="#a17Or6DtLG"
                        opacity="1"
                        fill-opacity="0"
                        stroke="rgba(255, 255, 255, 1.0)"
                        stroke-width="1"
                        stroke-opacity="1"
                    ></use>
                    </g>
                </g>
                <g>
                    <g>
                    <use
                        xlink:href="#j4QWBAJZw"
                        opacity="1"
                        fill-opacity="0"
                        stroke="rgba(255, 255, 255, 1.0)"
                        stroke-width="1"
                        stroke-opacity="1"
                    ></use>
                    </g>
                </g>
                <g>
                    <g>
                    <use
                        xlink:href="#bhjRTNplq"
                        opacity="1"
                        fill-opacity="0"
                        stroke="rgba(255, 255, 255, 1.0)"
                        stroke-width="1"
                        stroke-opacity="1"
                    ></use>
                    </g>
                </g>
                <g>
                    <g>
                    <use
                        xlink:href="#cUv46xKpz"
                        opacity="1"
                        fill-opacity="0"
                        stroke="rgba(255, 255, 255, 1.0)"
                        stroke-width="1"
                        stroke-opacity="1"
                    ></use>
                    </g>
                </g>
                <g>
                    <g>
                    <use
                        xlink:href="#a86zQgXCB"
                        opacity="1"
                        fill-opacity="0"
                        stroke="rgba(255, 255, 255, 1.0)"
                        stroke-width="1"
                        stroke-opacity="1"
                    ></use>
                    </g>
                </g>
                <g>
                    <g>
                    <use
                        xlink:href="#cf4JQTqTn"
                        opacity="1"
                        fill-opacity="0"
                        stroke="rgba(255, 255, 255, 1.0)"
                        stroke-width="1"
                        stroke-opacity="1"
                    ></use>
                    </g>
                </g>
                <g>
                    <g>
                    <use
                        xlink:href="#a2mhrsE0Kx"
                        opacity="1"
                        fill-opacity="0"
                        stroke="rgba(255, 255, 255, 1.0)"
                        stroke-width="1"
                        stroke-opacity="1"
                    ></use>
                    </g>
                </g>
                <g>
                    <g>
                    <use
                        xlink:href="#bmGTT9MeB"
                        opacity="1"
                        fill-opacity="0"
                        stroke="rgba(255, 255, 255, 1.0)"
                        stroke-width="1"
                        stroke-opacity="1"
                    ></use>
                    </g>
                </g>
                <g>
                    <g>
                    <use
                        xlink:href="#c5ckw7gUg"
                        opacity="1"
                        fill-opacity="0"
                        stroke="rgba(255, 255, 255, 1.0)"
                        stroke-width="1"
                        stroke-opacity="1"
                    ></use>
                    </g>
                </g>
                <g>
                    <g>
                    <use
                        xlink:href="#bNq0DnU5M"
                        opacity="1"
                        fill-opacity="0"
                        stroke="rgba(255, 255, 255, 1.0)"
                        stroke-width="1"
                        stroke-opacity="1"
                    ></use>
                    </g>
                </g>
                <g>
                    <g>
                    <use
                        xlink:href="#b1AsRfq0o"
                        opacity="1"
                        fill-opacity="0"
                        stroke="rgba(255, 255, 255, 1.0)"
                        stroke-width="1"
                        stroke-opacity="1"
                    ></use>
                    </g>
                </g>
                <g>
                    <g>
                    <use
                        xlink:href="#a5Fonrsblc"
                        opacity="1"
                        fill-opacity="0"
                        stroke="rgba(255, 255, 255, 1.0)"
                        stroke-width="1"
                        stroke-opacity="1"
                    ></use>
                    </g>
                </g>
                <g>
                    <g>
                    <use
                        xlink:href="#a6xLjyPrwa"
                        opacity="1"
                        fill-opacity="0"
                        stroke="rgba(255, 255, 255, 1.0)"
                        stroke-width="1"
                        stroke-opacity="1"
                    ></use>
                    </g>
                </g>
                <g>
                    <g>
                    <use
                        xlink:href="#feqWMbnGx"
                        opacity="1"
                        fill-opacity="0"
                        stroke="rgba(255, 255, 255, 1.0)"
                        stroke-width="1"
                        stroke-opacity="1"
                    ></use>
                    </g>
                </g>
                <g>
                    <g>
                    <use
                        xlink:href="#b877fx0LIT"
                        opacity="1"
                        fill-opacity="0"
                        stroke="rgba(255, 255, 255, 1.0)"
                        stroke-width="1"
                        stroke-opacity="1"
                    ></use>
                    </g>
                </g>
                <g>
                    <g>
                    <use
                        xlink:href="#aHbsBwJX"
                        opacity="1"
                        fill-opacity="0"
                        stroke="rgba(255, 255, 255, 1.0)"
                        stroke-width="1"
                        stroke-opacity="1"
                    ></use>
                    </g>
                </g>
                <g>
                    <g>
                    <use
                        xlink:href="#b7wp4SAvn"
                        opacity="1"
                        fill-opacity="0"
                        stroke="rgba(255, 255, 255, 1.0)"
                        stroke-width="1"
                        stroke-opacity="1"
                    ></use>
                    </g>
                </g>
                <g>
                    <g>
                    <use
                        xlink:href="#c1lg9Iyd5S"
                        opacity="1"
                        fill-opacity="0"
                        stroke="rgba(255, 255, 255, 1.0)"
                        stroke-width="1"
                        stroke-opacity="1"
                    ></use>
                    </g>
                </g>
                <g>
                    <g>
                    <use
                        xlink:href="#b531Gp81NK"
                        opacity="1"
                        fill-opacity="0"
                        stroke="rgba(255, 255, 255, 1.0)"
                        stroke-width="1"
                        stroke-opacity="1"
                    ></use>
                    </g>
                </g>
                <g>
                    <g>
                    <use
                        xlink:href="#c4gBuIj7W"
                        opacity="1"
                        fill-opacity="0"
                        stroke="rgba(255, 255, 255, 1.0)"
                        stroke-width="1"
                        stroke-opacity="1"
                    ></use>
                    </g>
                </g>
                <g>
                    <g>
                    <use
                        xlink:href="#g75dLLAPY"
                        opacity="1"
                        fill-opacity="0"
                        stroke="rgba(255, 255, 255, 1.0)"
                        stroke-width="1"
                        stroke-opacity="1"
                    ></use>
                    </g>
                </g>
                <g>
                    <g>
                    <use
                        xlink:href="#b9EtFKjjmd"
                        opacity="1"
                        fill-opacity="0"
                        stroke="rgba(255, 255, 255, 1.0)"
                        stroke-width="1"
                        stroke-opacity="1"
                    ></use>
                    </g>
                </g>
                </g>
            </g>
            </g>
        </svg>
        `;
  return nnlogo;
};

const webnnsamplenav = () => {
  const nnnav = `
        <li class="nav-item dropdown">
        <a class="nav-link dropdown-toggle" data-toggle="dropdown" role="button" aria-expanded="false">
            Samples
        </a>
        <div class="dropdown-menu" id="webnn-dropdown">
            <a class="dropdown-item" href="../image_classification/index.html">Image Classification</a>
            <a class="dropdown-item" href="../lenet/index.html">Handwritten Digits Classification</a>
            <a class="dropdown-item" href="../nsnet2/index.html">Noise Suppression (NSNet2)</a>
            <a class="dropdown-item" href="../rnnoise/index.html">Noise Suppression (RNNoise)</a>
            <a class="dropdown-item" href="../style_transfer/index.html">Fast Style Transfer</a>
            <a class="dropdown-item" href="../object_detection/index.html">Object Detection</a>
            <a class="dropdown-item" href="../semantic_segmentation/index.html">Semantic Segmentation</a>
            <a class="dropdown-item" href="../face_recognition/index.html">Face Recognition</a>
            <a class="dropdown-item" href="../facial_landmark_detection/index.html">Facial landmark Detection</a>
            <a class="dropdown-item" href="../code/index.html">WebNN Code Editor</a>
        </div>
        </li>
    `;
  return nnnav;
};

const webnnbadge = () => {
  const nnbadge = `
            <div class='webnnbadge mb-4'>
                <div class='webnn-title'>WebNN API</div>
                <div id="webnnstatus"></div>
            </div>
            <div class='webnnbadge mb-4'>
                <div class='webnn-title'>W3C Spec</div>
                <div class='webnn-status-true'><a href='https://www.w3.org/TR/webnn/#usecases'
                    title='W3C Web Neural Network API Use Cases'>Use Cases</a></div>
            </div>
        `;
  return nnbadge;
};

const footer = () => {
    const footerlink = `
        <p>
          The WebNN API is under active development within W3C Web Machine Learning Working Group, please <a href="https://github.com/webmachinelearning/webnn-samples/issues" title="File a bug report for WebNN Samples">file a bug report</a> if the WebNN sample doesn't work in the latest versions of Chrome or Edge.
        </p>
        <p>
          &copy;2024 
          <a href="https://webmachinelearning.github.io/">WebNN API</a> ·
          <a href="https://github.com/webmachinelearning/webnn-samples#webnn-installation-guides">Installation Guides</a> · 
          <a href="https://webmachinelearning.github.io/webnn-status/">Implementation Status</a>
        </p>
    `;
    return footerlink;
}

$(document).ready(async () => {
  $("nav ul.navbar-nav").html(webnnsamplenav());
  $("#logosvg").html(webnnlogo());
  $("#badge").html(webnnbadge());
  $("#footer").html(footer());
  if (await isWebNN()) {
    if ($("#backendBtns")) {
      if (!isElectron()) {
        $('label[name="polyfill"]').addClass("disabled");
        $('label[name="polyfill"]').addClass("btn-outline-secondary");
        $('label[name="polyfill"]').removeClass("btn-outline-info");
        $('label[name="polyfill"]').attr(
          "title",
          "WebNN is supported, disable WebNN Polyfill."
        );
      }
    }
    $("#webnnstatus").html("supported").addClass("webnn-status-true");
  } else {
    if ($("#backendBtns")) {
      $('label[name="webnn"]').addClass("disabled");
      $('label[name="webnn"]').addClass("btn-outline-secondary");
      $('label[name="webnn"]').removeClass("btn-outline-info");
      $('label[name="webnn"]').attr("title", "WebNN is not supported!");
    }
    $("#webnnstatus").html("not supported").addClass("webnn-status-false");
  }
});
