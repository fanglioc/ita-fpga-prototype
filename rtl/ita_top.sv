`timescale 1ns / 1ps

module ita_top #(
    parameter INPUT_DIM = 64,
    parameter HIDDEN_DIM = 128,
    parameter OUTPUT_DIM = 64,
    parameter WEIGHT_BITS = 4,
    parameter ACT_BITS = 8
)(
    input  logic clk,
    input  logic rst_btn_n,
    output logic uart_tx,
    output logic [3:0] led,
    input  logic [3:0] sw
);

    logic rst_n;
    logic [2:0] rst_sync;
    
    always_ff @(posedge clk) begin
        rst_sync <= {rst_sync[1:0], rst_btn_n};
        rst_n <= rst_sync[2];
    end
    
    // SIMPLIFIED Test Pattern Generator - just level trigger
    logic [ACT_BITS-1:0] test_input [INPUT_DIM];
    logic test_input_valid;
    logic test_input_ready;
    logic test_done;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            test_input_valid <= 0;
            test_done <= 0;
            // Pre-fill test data
            for (int i = 0; i < INPUT_DIM; i++) begin
                test_input[i] <= {i[5:0], 2'b01};
            end
        end else if (sw[0] && !test_done && !test_input_valid) begin
            // SW0 is high and we haven't run yet
            test_input_valid <= 1;
        end else if (test_input_valid && test_input_ready) begin
            // Expand accepted the data
            test_input_valid <= 0;
            test_done <= 1;
        end
    end
    
    // Pipeline
    logic [ACT_BITS-1:0] expand_out [HIDDEN_DIM];
    logic expand_valid;
    
    logic [ACT_BITS-1:0] contract_out [OUTPUT_DIM];
    logic contract_valid;
    
    expand_engine #(
        .INPUT_DIM(INPUT_DIM),
        .OUTPUT_DIM(HIDDEN_DIM),
        .WEIGHT_BITS(WEIGHT_BITS),
        .ACT_BITS(ACT_BITS)
    ) u_expand (
        .clk(clk),
        .rst_n(rst_n),
        .input_vec(test_input),
        .input_valid(test_input_valid),
        .input_ready(test_input_ready),
        .output_vec(expand_out),
        .output_valid(expand_valid)
    );
    
    contract_engine #(
        .INPUT_DIM(HIDDEN_DIM),
        .OUTPUT_DIM(OUTPUT_DIM),
        .WEIGHT_BITS(WEIGHT_BITS),
        .ACT_BITS(ACT_BITS)
    ) u_contract (
        .clk(clk),
        .rst_n(rst_n),
        .input_vec(expand_out),
        .input_valid(expand_valid),
        .output_vec(contract_out),
        .output_valid(contract_valid)
    );
    
    // UART Output
    logic [$clog2(OUTPUT_DIM)-1:0] uart_cnt;
    logic uart_busy;
    logic [7:0] uart_data;
    logic uart_send;
    logic uart_sending;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            uart_cnt <= 0;
            uart_send <= 0;
            uart_sending <= 0;
        end else if (contract_valid && !uart_sending) begin
            uart_sending <= 1;
            uart_cnt <= 0;
            uart_data <= contract_out[0];
            uart_send <= 1;
        end else if (uart_sending) begin
            if (!uart_busy && uart_send) begin
                uart_send <= 0;
            end else if (!uart_busy && !uart_send) begin
                if (uart_cnt < OUTPUT_DIM - 1) begin
                    uart_cnt <= uart_cnt + 1;
                    uart_data <= contract_out[uart_cnt + 1];
                    uart_send <= 1;
                end else begin
                    uart_sending <= 0;
                end
            end
        end
    end
    
    uart_tx #(
        .CLK_FREQ(125_000_000),
        .BAUD_RATE(115200)
    ) u_uart (
        .clk(clk),
        .rst_n(rst_n),
        .tx_data(uart_data),
        .tx_valid(uart_send),
        .tx_busy(uart_busy),
        .tx(uart_tx)
    );
    
    // LED indicators
    assign led[0] = test_input_valid || test_done;  // ON when test runs or completed
    assign led[1] = expand_valid;
    assign led[2] = contract_valid;
    assign led[3] = uart_busy;
    
endmodule
