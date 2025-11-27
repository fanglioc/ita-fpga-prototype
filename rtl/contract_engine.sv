`timescale 1ns / 1ps

module contract_engine #(
    parameter INPUT_DIM = 128,
    parameter OUTPUT_DIM = 64,
    parameter WEIGHT_BITS = 4,
    parameter ACT_BITS = 8
)(
    input  logic clk,
    input  logic rst_n,
    input  logic [ACT_BITS-1:0] input_vec [INPUT_DIM],
    input  logic input_valid,
    output logic [ACT_BITS-1:0] output_vec [OUTPUT_DIM],
    output logic output_valid
);

    // [Same structure as expand_engine]
    // Key differences:
    // 1. INPUT_DIM = 128, OUTPUT_DIM = 64
    // 2. Weight file: weights_contract.hex
    
    typedef enum logic [1:0] {IDLE, COMPUTE, DONE} state_t;
    state_t state;
    
    logic [WEIGHT_BITS-1:0] weights [OUTPUT_DIM][INPUT_DIM];
    initial $readmemh("../weights/weights_contract.hex", weights);
    
    logic [$clog2(INPUT_DIM):0] compute_cnt;
    logic [ACT_BITS+WEIGHT_BITS+$clog2(INPUT_DIM)-1:0] accumulators [OUTPUT_DIM];
    logic [ACT_BITS+WEIGHT_BITS-1:0] mult_results [OUTPUT_DIM];
    
    genvar i;
    generate
        for (i = 0; i < OUTPUT_DIM; i++) begin : mac_array
            assign mult_results[i] = input_vec[compute_cnt] * weights[i][compute_cnt];
            
            always_ff @(posedge clk) begin
                if (state == IDLE) accumulators[i] <= 0;
                else if (state == COMPUTE) accumulators[i] <= accumulators[i] + mult_results[i];
            end
            
            always_ff @(posedge clk) begin
                if (state == DONE) begin
                    if (accumulators[i][ACT_BITS+WEIGHT_BITS+$clog2(INPUT_DIM)-1]) begin
                        output_vec[i] <= 0;
                    end else begin
                        logic [ACT_BITS+WEIGHT_BITS+$clog2(INPUT_DIM)-1:0] shifted;
                        shifted = accumulators[i] >> WEIGHT_BITS;
                        output_vec[i] <= (shifted > 255) ? 255 : shifted[ACT_BITS-1:0];
                    end
                end
            end
        end
    endgenerate
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            compute_cnt <= 0;
            output_valid <= 0;
        end else begin
            case (state)
                IDLE: begin
                    output_valid <= 0;
                    if (input_valid) begin
                        state <= COMPUTE;
                        compute_cnt <= 0;
                    end
                end
                COMPUTE: begin
                    compute_cnt <= compute_cnt + 1;
                    if (compute_cnt == INPUT_DIM - 1) state <= DONE;
                end
                DONE: begin
                    output_valid <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end
    
endmodule
